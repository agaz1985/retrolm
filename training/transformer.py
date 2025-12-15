import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------
# Configuration
# ----------------------------
SEQ_LEN = 8       # Sequence length
VOCAB_SIZE = 256  # ASCII
EMBED_DIM = 16    # Embedding size
FF_DIM = 32       # Feed-forward hidden size
NUM_HEADS = 1     # Attention heads
NUM_LAYERS = 1    # Transformer layers

# ----------------------------
# Transformer
# ----------------------------
class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embed = nn.Embedding(VOCAB_SIZE, EMBED_DIM)
        self.pos_embed = nn.Parameter(torch.randn(SEQ_LEN, EMBED_DIM) * 0.1)

        # Attention linear projections
        self.Wq = nn.Linear(EMBED_DIM, EMBED_DIM, bias=True)
        self.Wk = nn.Linear(EMBED_DIM, EMBED_DIM, bias=True)
        self.Wv = nn.Linear(EMBED_DIM, EMBED_DIM, bias=True)
        self.Wo = nn.Linear(EMBED_DIM, EMBED_DIM, bias=True)

        # Feed-forward
        self.W1 = nn.Linear(EMBED_DIM, FF_DIM)
        self.W2 = nn.Linear(FF_DIM, EMBED_DIM)

        # LM head
        self.lm_head = nn.Linear(EMBED_DIM, VOCAB_SIZE, bias=True)

    def forward(self, x):
        B, N = x.shape
        X = self.token_embed(x) + self.pos_embed[:N]  # [B, seq, embed]

        # --- Self-Attention ---
        Q = self.Wq(X)
        K = self.Wk(X)
        V = self.Wv(X)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (EMBED_DIM ** 0.5)
        weights = F.softmax(scores, dim=-1)
        attention_out = torch.matmul(weights, V)
        X = X + self.Wo(attention_out)  # residual

        # --- Feed-Forward ---
        FF = F.relu(self.W1(X))
        X = X + self.W2(FF)  # residual

        # --- LM head ---
        logits = self.lm_head(X)
        return logits
