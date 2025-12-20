import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from config import ModelConfig

class RetroLLMTransformer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        
        self.config = config
        
        # Embeddings
        self.token_embed = nn.Embedding(config.vocab_size, config.embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(config.seq_len, config.embed_dim) * 0.02)
        
        # Dropout layers
        self.embed_dropout = nn.Dropout(config.dropout)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.ff_dropout = nn.Dropout(config.dropout)
        
        # Single-head attention
        self.Wq = nn.Linear(config.embed_dim, config.embed_dim, bias=True)
        self.Wk = nn.Linear(config.embed_dim, config.embed_dim, bias=True)
        self.Wv = nn.Linear(config.embed_dim, config.embed_dim, bias=True)
        self.Wo = nn.Linear(config.embed_dim, config.embed_dim, bias=True)
        
        # Feed-forward
        self.W1 = nn.Linear(config.embed_dim, config.ff_dim, bias=True)
        self.W2 = nn.Linear(config.ff_dim, config.embed_dim, bias=True)
        
        # LM head (weight tied)
        self.lm_head = nn.Linear(config.embed_dim, config.vocab_size, bias=True)
        self.lm_head.weight = self.token_embed.weight
        
        self.scale = config.embed_dim ** 0.5
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        
    def forward(self, x):
        B, N = x.shape        
        X = self.token_embed(x) + self.pos_embed[:N]
        X = self.embed_dropout(X)
        
        # Self-Attention
        Q = self.Wq(X)
        K = self.Wk(X)
        V = self.Wv(X)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        causal_mask = torch.triu(torch.ones(N, N, device=x.device), diagonal=1).bool()
        scores = scores.masked_fill(causal_mask, float('-inf'))
        
        weights = F.softmax(scores, dim=-1)
        weights = self.attn_dropout(weights)  # Dropout on attention weights
        
        attention_out = torch.matmul(weights, V)
        attention_out = self.Wo(attention_out)
        attention_out = self.resid_dropout(attention_out)
        
        X = X + attention_out
        
        # Feed-Forward
        FF = F.relu(self.W1(X))
        FF = self.ff_dropout(FF)  # Dropout after activation
        FF = self.W2(FF)
        FF = self.resid_dropout(FF)
        
        X = X + FF
        
        # Output
        logits = self.lm_head(X)
        return logits
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        self.eval()
        
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.seq_len else idx[:, -self.config.seq_len:]
            logits = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)
        
        return idx
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())

# Optimizer setup
def create_optimizer(model, config):
    """Create optimizer with proper weight decay separation"""
    decay_params = []
    no_decay_params = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            # Don't apply weight decay to:
            # - biases
            # - positional embeddings
            # - lm_head (it's tied to token_embed, so we'd double-count)
            if 'bias' in name or 'pos_embed' in name or 'lm_head' in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)
    
    optimizer_groups = [
        {'params': decay_params, 'weight_decay': config.weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0}
    ]
    
    optimizer = torch.optim.AdamW(
        optimizer_groups,
        lr=config.learning_rate,
        betas=(0.9, 0.95),
        eps=1e-8
    )
    
    return optimizer


# Learning rate scheduler with warmup and cosine decay
def get_lr_scheduler(optimizer, config, total_steps):
    """Create learning rate scheduler with warmup and cosine decay"""
    def lr_lambda(step):
        # Warmup phase
        if step < config.warmup_steps:
            return step / max(1, config.warmup_steps)
        
        # Cosine decay with minimum LR
        # Avoid division by zero if total_steps <= warmup_steps
        decay_steps = max(1, total_steps - config.warmup_steps)
        progress = (step - config.warmup_steps) / decay_steps
        progress = min(1.0, progress)  # Clamp to [0, 1]
        min_lr_ratio = 0.1  # Don't decay below 10% of peak LR
        cosine_decay = 0.5 * (1 + math.cos(progress * math.pi))
        return min_lr_ratio + (1 - min_lr_ratio) * cosine_decay
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return scheduler