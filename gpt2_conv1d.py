from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import requests
import tiktoken
import os 
import time
import math
import inspect
from datasets import load_dataset
from torch.utils.tensorboard import SummaryWriter


### ======== SETUP ALL HYPERPARAMS HERE ========
hyperparams = {
    'batch_size': 32,
    'seq_length': 1024,
    'max_lr': 2e-4,
    'min_lr': 1e-5,
    'warmup_steps': 50,
    'eval_steps': 50,
    'max_steps': 1500,
    'weight_decay': 0.1,
    # 'total_batch_size': 524288,
    'total_batch_size': 32768,
    'vocab_size': 50304,
    'kernel_size': 2,
    'stride': 2,
}

tensorboard_logging = True

train_cache_path = '/raid/dmerkulov/datasets/text/tinystories/train_tokenized_cache.pt'
valid_cache_path = '/raid/dmerkulov/datasets/text/tinystories/valid_tokenized_cache.pt'

locals().update(hyperparams)

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"


### ======== MODEL DEFINITION ========

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        # to split embeddings dimension across attention heads
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3*config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # not really a 'bias', more of a mask, but following the OpenAI/HF naming
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1, 1, config.block_size, config.block_size))
        
    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be 
        # nh is "number of heads", hs is "head size" and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M) n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # this is flash-attention
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        # output projection
        y = self.c_proj(y)
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4*config.n_embd)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4*config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024 # max sequence length
    vocab_size: int = 50257 # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 12 # number of layers
    n_head: int = 12 # number of heads
    n_embd: int = 768 # embedding dimension
    kernel_size: int = 0 # Affects the speed and memory
    stride: int = 0 # Affects the reduction

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.conv_mode = False
        if (config.kernel_size > 0) and (config.stride > 0):
            self.conv_mode = True

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h  = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
            ))
        if self.conv_mode:
            self.conv = nn.Conv1d(
                in_channels=config.n_embd, 
                out_channels=config.n_embd, 
                kernel_size=config.kernel_size, 
                stride=config.stride)
            self.convt = nn.ConvTranspose1d(
                in_channels=config.n_embd, 
                out_channels=config.n_embd, 
                kernel_size=config.kernel_size, 
                stride=config.stride)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight

        # init params
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        # idx is of shape (B, T)
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # forward the token and posisition embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)
        x = tok_emb + pos_emb
        if self.conv_mode:
            x = x.transpose(1, 2)
            x = self.conv(x)
            x = x.transpose(1, 2)
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)

        if self.conv_mode:
            x = x.transpose(1, 2)
            x = self.convt(x)
            x = x.transpose(1, 2)
        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss
    
    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model
    
    def configure_optimizers(self, weight_decay, learning_rate, device_type):
        # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer

# attempt to autodetect device
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"

print(f"using device: {device}")

### ======== DATA LOADING ========
# Load TinyStories dataset from Hugging Face
dataset = load_dataset("roneneldan/TinyStories", split="train")
valid_dataset = load_dataset("roneneldan/TinyStories", split="validation")

# Tokenizer
enc = tiktoken.get_encoding('gpt2')

class DataLoaderLite:
    def __init__(self, dataset, B, T, cache_path=None):
        self.dataset = dataset
        self.B = B
        self.T = T
        self.cache_path = cache_path
        self.tokens = self._load_or_tokenize_data()
        print(f"Loaded {len(self.tokens)} tokens")
        print(f"1 epoch = {len(self.tokens) // (B*T)} batches")

        # state
        self.current_position = 0

    def _load_or_tokenize_data(self):
        if self.cache_path and os.path.exists(self.cache_path):
            print(f"Loading tokenized data from cache: {self.cache_path}")
            tokens = torch.load(self.cache_path)
        else:
            text = " ".join(self.dataset['text'])  # Combine all texts into one string
            tokens = enc.encode(text)
            tokens = torch.tensor(tokens)
            if self.cache_path:
                print(f"Saving tokenized data to cache: {self.cache_path}")
                torch.save(tokens, self.cache_path)
        return tokens

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance the position in the tensor
        self.current_position += B * T
        # if loading the next batch would be out of bounds, advance to next shard
        if self.current_position + (B * T + 1) > len(self.tokens):
            self.current_position = 0
        return x, y

### ======== Optimization ========

torch.manual_seed(228)
if torch.cuda.is_available():
    torch.cuda.manual_seed(228)

assert total_batch_size % (batch_size*seq_length) == 0, "make sure total_batch_size is divisible by B*T"
grad_accum_steps = total_batch_size // (batch_size*seq_length)
print(f"total desired batch size: {total_batch_size}")
print(f"calculated gradient accumulation steps: {grad_accum_steps}")

train_loader = DataLoaderLite(dataset, B=batch_size, T=seq_length, cache_path=train_cache_path)
valid_loader = DataLoaderLite(valid_dataset, B=batch_size, T=seq_length, cache_path=valid_cache_path)

print("data loaded")

torch.set_float32_matmul_precision('high')
model = GPT(GPTConfig(vocab_size=vocab_size, kernel_size=kernel_size, stride=stride))
model.to(device)

print("model loaded")

compile_model = False
if compile_model:
    model = torch.compile(model)
    print(f"model compiled")
else:
    print(f"model not compiled")

def get_lr(it):
    # warmup
    if it < warmup_steps:
        return max_lr*(it+1)/warmup_steps
    # min lr at the end
    if it > max_steps:
        return min_lr
    # cosine decay
    decay_ratio = (it - warmup_steps)/(max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi*decay_ratio)) # coeff starts at 1 and goes to 0
    return min_lr + coeff*(max_lr - min_lr)

optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=max_lr, device_type=device)

### INITIALIZE TENSORBOARD FOR LOGGING
# Initialize TensorBoard
def make_run_name(hyperparams: dict) -> str:
    return "_".join(f"{key}_{value}" for key, value in hyperparams.items())

log_dir = make_run_name(hyperparams)  # Change the directory name as needed
if tensorboard_logging:
    writer = SummaryWriter(log_dir=f"runs/{log_dir}")

for step in range(max_steps):
    t0 = time.time()
    optimizer.zero_grad()
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, loss = model(x, y)
        loss = loss/grad_accum_steps
        loss_accum += loss.detach()
        loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    optimizer.step()
    if device == "cuda":
        torch.cuda.synchronize()
    t1 = time.time()
    dt = (t1 - t0)*1000 # milliseconds
    processed_tokens = train_loader.B*train_loader.T*grad_accum_steps
    tok_per_sec = processed_tokens/(t1- t0)

    # Log metrics to TensorBoard
    if tensorboard_logging:
        writer.add_scalar('Loss/train', loss_accum, step)
        writer.add_scalar('Learning Rate', lr, step)
        writer.add_scalar('Gradient Norm', norm, step)
        writer.add_scalar('Tokens Per Second', tok_per_sec, step)
        writer.add_scalar('Time Per Step (ms)', dt, step)

    print(f"Step {step:3d} | Epoch {processed_tokens*(step+1)/len(train_loader.tokens):.3f} | loss: {loss_accum:.3f} | 🔽 norm {norm:.2f} | lr {lr:.5f} | ⏰: {dt:.2f}ms, ⚡️: {tok_per_sec:.0f} tok/sec")

    # Validation Loss Calculation
    if step % eval_steps == 0 or step==max_steps-1:
        val_loss_accum = 0.0
        model.eval()
        with torch.no_grad():
            for i in range(grad_accum_steps):
                x, y = valid_loader.next_batch()
                x, y = x.to(device), y.to(device)
                logits, loss = model(x, y)
                val_loss_accum += loss.detach()
        model.train()
        val_loss_accum /= grad_accum_steps
        if tensorboard_logging:
            writer.add_scalar('Loss/validation', val_loss_accum, step)
        print(f"Validation Loss at step {step}: {val_loss_accum:.3f}")

        # Sampling from the model
        tokens = enc.encode("A long time ago in a galaxy far far away ")
        batch_size = 4
        tokens = torch.tensor(tokens, dtype=torch.long)
        tokens = tokens.unsqueeze(0).repeat(batch_size, 1)
        x = tokens.to(device)

        max_length = 40
        torch.manual_seed(228)
        while x.size(1) < max_length:
            logits, loss = model(x)
            # take the logits at the last position
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            # do top-k sampling of 10
            topk_probs, topk_indices = torch.topk(probs, 10, dim=-1)
            # select a token from the top-k probabilities
            ix = torch.multinomial(topk_probs, 1)
            # gather the corresponding indices
            xcol = torch.gather(topk_indices, -1, ix)
            # append to the sequence
            x = torch.cat((x, xcol), dim=1)

        # print the generated text
        for i in range(batch_size):
            tokens = x[i, :max_length].tolist()
            decoded = enc.decode(tokens)
            print("🤖 Generated text:", decoded)
            if tensorboard_logging:
                writer.add_text('Generated Text', decoded, step-batch_size+i)

if tensorboard_logging:
    writer.add_hparams(
        hyperparams,
        {
            "Loss/train": loss_accum,
            "Loss/validation": val_loss_accum
        },
        )