"""
This code is a proof of concept for fine-tuning gpt-j with limited memory.
The dataset used is an adaptation of Alpaca dataset in order to,
turn a text generation model into a instruct model.
Dataset: https://huggingface.co/datasets/nlpcloud/instructions-dataset-adapted-from-stanford-alpaca-for-gpt-j
"""

from loguru import logger
import transformers
from transformers import AutoTokenizer, TFAutoModel
import torch
import torch.nn.functional as F
from torch import nn
from torch.cuda.amp import custom_fwd, custom_bwd
from bitsandbytes.functional import quantize_blockwise, dequantize_blockwise
from tqdm.auto import tqdm
from datasets import load_dataset
from bitsandbytes.optim import Adam8bit
import time, os

class FrozenBNBLinear(nn.Module):
    # constructor function for a PyTorch module.
    def __init__(self, weight, absmax, code, bias=None): 
        assert isinstance(bias, nn.Parameter) or bias is None
        super().__init__()
        self.out_features, self.in_features = weight.shape
        self.register_buffer("weight", weight.requires_grad_(False)) #register the tensors as buffers
        self.register_buffer("absmax", absmax.requires_grad_(False))
        self.register_buffer("code", code.requires_grad_(False))
        self.adapter = None
        self.bias = bias

    def forward(self, input):
        output = DequantizeAndLinear.apply(input, self.weight, self.absmax, self.code, self.bias) #converts a quantized tensor to floating point tensor (performs linear transformation)
        if self.adapter:
            output += self.adapter(input)
        return output #returns the linear transformation
    
    @classmethod
    def from_linear(cls, linear: nn.Linear) -> "FrozenBNBLinear":
        weights_int8, state = quantize_blockise_lowmemory(linear.weight)
        return cls(weights_int8, *state, linear.bias)
    
    def __repr__(self):
        return f"{self.__class__.__name__}({self.in_features}, {self.out_features})"
    

class DequantizeAndLinear(torch.autograd.Function): 
    @staticmethod
    @custom_fwd
    def forward(ctx, input: torch.Tensor, weights_quantized: torch.ByteTensor, absmax: torch.FloatTensor, code: torch.FloatTensor, bias: torch.FloatTensor):
        weights_deq = dequantize_blockwise(weights_quantized, absmax=absmax, code=code) #to dequantize the weights tensor
        ctx.save_for_backward(input, weights_quantized, absmax, code)
        ctx._has_bias = bias is not None
        return F.linear(input, weights_deq, bias).clone() #returns the result of linear transformation on i/p tensor using weights_deq and bias
    
    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output: torch.Tensor): #calculate the gradients of the inputs
        assert not ctx.needs_input_grad[1] and not ctx.needs_input_grad[2] and not ctx.needs_input_grad[3]
        input, weights_quantized, absmax, code = ctx.saved_tensors
        weights_deq = dequantize_blockwise(weights_quantized, absmax=absmax, code=code) 
        grad_input = grad_output @ weights_deq 
        grad_bias = grad_output.flatten(0, -2).sum(dim=0) if ctx._has_bias else None
        return grad_input, None, None, None, grad_bias
    

class FrozenBNBEmbedding(nn.Module):
    #The embedding layer takes an input of indices, performs a lookup in the weight matrix, and returns the corresponding embedding vectors.
    def __init__(self, weight, absmax, code):
        super().__init__()
        self.num_embeddings, self.embedding_dim = weight.shape
        self.register_buffer("weight", weight.requires_grad_(False))
        self.register_buffer("absmax", absmax.requires_grad_(False))
        self.register_buffer("code", code.requires_grad_(False))
        self.adapter = None

    def forward(self, input, **kwargs):
        with torch.no_grad():
            weight_deq = dequantize_blockwise(self.weight, absmax=self.absmax, code=self.code)
            output = F.embedding(input, weight_deq, **kwargs)
        if self.adapter:
            output += self.adapter(input)
        return output

    @classmethod
    def from_embedding(cls, embedding: nn.Embedding) -> "FrozenBNBEmbedding": #creates a new instance of FrozenBNBEmbedding
        weights_int8, state = quantize_blockise_lowmemory(embedding.weight)
        return cls(weights_int8, *state)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.num_embeddings}, {self.embedding_dim})"
    

def quantize_blockise_lowmemory(matrix: torch.Tensor, chunk_size: int = 2 ** 20): #quantize a given matrix into 8-bit integers in a block-wise manner, with a low memory footprint.
    assert chunk_size % 4096 ==0 #chunk size must be a multiple of 4096
    code = None
    chunks = []
    absmaxes = []
    flat_tensor = matrix.view(-1) #flattens to 1D
    for i in range((matrix.numel() - 1) // chunk_size + 1):
        input_chunk = flat_tensor[i * chunk_size: (i + 1) * chunk_size].clone()
        quantized_chunk, (absmax_chunk, code) = quantize_blockwise(input_chunk, code=code)
        chunks.append(quantized_chunk)
        absmaxes.append(absmax_chunk)
    matrix_i8 = torch.cat(chunks).reshape_as(matrix)
    absmax = torch.cat(absmaxes)
    return matrix_i8, (absmax, code)

def convert_to_int8(model): #Convert linear and embedding modules to 8-bit with optional adpaters.
    for module in list(model.modules()):
        for name, child in module.named_children():
            if isinstance(child, nn.Linear):
                print(name, child)
                setattr(
                    module,
                    name,
                    FrozenBNBLinear(
                        weight=torch.zeros(child.out_features, child.in_features, dtype=torch.uint8),
                        absmax=torch.zeros((child.weight.numel() - 1) // 4096 + 1),
                        code=torch.zeros(256),
                        bias=child.bias,
                    ),
                )
            elif isinstance(child, nn.Embedding):
                setattr(
                    module,
                    name,
                    FrozenBNBEmbedding(
                        weight=torch.zeros(child.num_embeddings, child.embedding_dim, dtype=torch.uint8),
                        absmax=torch.zeros((child.weight.numel() - 1) // 4096 + 1),
                        code=torch.zeros(256),
                    )
                )

class GPTJBlock(transformers.models.gptj.modeling_gptj.GPTJBlock):
    def __init__(self, config):
        super().__init__(config)

        convert_to_int8(self.attn)
        convert_to_int8(self.mlp)


class GPTJModel(transformers.models.gptj.modeling_gptj.GPTJModel):
    def __init__(self, config):
        super().__init__(config)
        convert_to_int8(self)
        

class GPTJForCausalLM(transformers.models.gptj.modeling_gptj.GPTJForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        convert_to_int8(self)


transformers.models.gptj.modeling_gptj.GPTJBlock = GPTJBlock 

model_name = "EleutherAI/gpt-j-6B"

config = transformers.GPTJConfig.from_pretrained(model_name) #loading model
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
gpt = GPTJForCausalLM.from_pretrained("hivemind/gpt-j-6B-8bit", low_cpu_mem_usage=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
gpt.to(device)

#lora fine tune
def add_adapters(model, adapter_dim=16):
    assert adapter_dim > 0

    for module in model.modules():
        if isinstance(module, FrozenBNBLinear):
            module.adapter = nn.Sequential(
                nn.Linear(module.in_features, adapter_dim, bias=False),
                nn.Linear(adapter_dim, module.out_features, bias=False),
            )
            nn.init.zeros_(module.adapter[1].weight)
        elif isinstance(module, FrozenBNBEmbedding):
            module.adapter = nn.Sequential(
                nn.Embedding(module.num_embeddings, adapter_dim),
                nn.Linear(adapter_dim, module.embedding_dim, bias=False),
            )
            nn.init.zeros_(module.adapter[1].weight)

add_adapters(gpt)
gpt.to(device)
gpt.gradient_checkpointing_enable()

dataset = load_dataset("nlpcloud/instructions-dataset-adapted-from-stanford-alpaca-for-gpt-j", streaming=True)
#custom dataset
#dataset = load_dataset('text', data_files={'train': ['article-1.txt', 'article-2.txt'], 'test': ['article-3.txt', 'article-4.txt']})
optimizer = Adam8bit(gpt.parameters(), lr=1e-5)
start = time.time()

#train loop
with torch.cuda.amp.autocast():
    for row in tqdm(dataset["train"]):
        if len(row["text"]) <= 1:
            continue

        batch = tokenizer(row["text"], truncation=True, max_length=128, return_tensors='pt')
        batch = {k: v.cuda() for k, v in batch.items()}

        out = gpt.forward(**batch,)

        loss = F.cross_entropy(out.logits[:, :-1, :].flatten(0, -2), batch['input_ids'][:, 1:].flatten(),
                               reduction='mean')
        print(loss)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

logger.info("Finished fine-tuning in {}".format(time.time() - start))
try:
    save_dir = "./gpt-j-6B" #save model here
    os.makedirs(save_dir)
    gpt.save_pretrained(save_dir)
except Exception as e:
    logger.info("Error saving model: {}".format(e))
