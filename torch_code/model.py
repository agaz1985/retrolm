import torch
import torch.nn as nn
import torch.nn.functional as F

# Configuration
SEQ_LEN = 8
VOCAB_SIZE = 128
EMBED_DIM = 12
FF_DIM = 24
NUM_LAYERS = 1

class RetroLLMTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Embeddings
        self.token_embed = nn.Embedding(VOCAB_SIZE, EMBED_DIM)
        self.pos_embed = nn.Parameter(torch.randn(SEQ_LEN, EMBED_DIM) * 0.02)
        
        # Single-head attention
        self.Wq = nn.Linear(EMBED_DIM, EMBED_DIM, bias=True)
        self.Wk = nn.Linear(EMBED_DIM, EMBED_DIM, bias=True)
        self.Wv = nn.Linear(EMBED_DIM, EMBED_DIM, bias=True)
        self.Wo = nn.Linear(EMBED_DIM, EMBED_DIM, bias=True)
        
        # Feed-forward
        self.W1 = nn.Linear(EMBED_DIM, FF_DIM, bias=True)
        self.W2 = nn.Linear(FF_DIM, EMBED_DIM, bias=True)
        
        # LM head (weight tied)
        self.lm_head = nn.Linear(EMBED_DIM, VOCAB_SIZE, bias=True)
        self.lm_head.weight = self.token_embed.weight
        
        self.scale = EMBED_DIM ** 0.5
        
    def forward(self, x):
        B, N = x.shape        
        X = self.token_embed(x) + self.pos_embed[:N]
        
        # Self-Attention
        Q = self.Wq(X)
        K = self.Wk(X)
        V = self.Wv(X)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        causal_mask = torch.triu(torch.ones(N, N, device=x.device), diagonal=1).bool()
        scores = scores.masked_fill(causal_mask, float('-inf'))
        
        weights = F.softmax(scores, dim=-1)
        attention_out = torch.matmul(weights, V)
        X = X + self.Wo(attention_out)
        
        # Feed-Forward
        FF = F.relu(self.W1(X))
        X = X + self.W2(FF)
        
        # Output
        logits = self.lm_head(X)
        return logits
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        self.eval()
        
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= SEQ_LEN else idx[:, -SEQ_LEN:]
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