import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------
# Configuration for Pentium III (128MB RAM)
# ----------------------------
SEQ_LEN = 16      # Increased from 8 for better context
VOCAB_SIZE = 256  # ASCII
EMBED_DIM = 16    # Keep small for P3
FF_DIM = 32       # 2x embed_dim
NUM_HEADS = 1     # Single head - simpler for C, faster on P3
NUM_LAYERS = 1    # Single layer - enough for tiny model
DROPOUT = 0.0     # No dropout for inference

# ----------------------------
# Pentium III Optimized Transformer
# ----------------------------
class TinyTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Embeddings
        self.token_embed = nn.Embedding(VOCAB_SIZE, EMBED_DIM)
        self.pos_embed = nn.Parameter(torch.randn(SEQ_LEN, EMBED_DIM) * 0.02)
        
        # Attention projections (keep bias for better quality)
        self.Wq = nn.Linear(EMBED_DIM, EMBED_DIM, bias=True)
        self.Wk = nn.Linear(EMBED_DIM, EMBED_DIM, bias=True)
        self.Wv = nn.Linear(EMBED_DIM, EMBED_DIM, bias=True)
        self.Wo = nn.Linear(EMBED_DIM, EMBED_DIM, bias=True)
        
        # Layer norms for stability
        self.ln1 = nn.LayerNorm(EMBED_DIM)
        self.ln2 = nn.LayerNorm(EMBED_DIM)
        
        # Feed-forward
        self.W1 = nn.Linear(EMBED_DIM, FF_DIM, bias=True)
        self.W2 = nn.Linear(FF_DIM, EMBED_DIM, bias=True)
        
        # Final layer norm
        self.ln_f = nn.LayerNorm(EMBED_DIM)
        
        # LM head (weight tied to save memory)
        self.lm_head = nn.Linear(EMBED_DIM, VOCAB_SIZE, bias=False)
        self.lm_head.weight = self.token_embed.weight  # Weight tying
        
        self.scale = EMBED_DIM ** 0.5
        
    def forward(self, x):
        B, N = x.shape        
        X = self.token_embed(x) + self.pos_embed[:N]  # [B, N, EMBED_DIM]
        
        # --- Self-Attention with Pre-LN ---
        X_norm = self.ln1(X)
        Q = self.Wq(X_norm)
        K = self.Wk(X_norm)
        V = self.Wv(X_norm)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # CRITICAL: Causal mask for language modeling
        causal_mask = torch.triu(torch.ones(N, N, device=x.device), diagonal=1).bool()
        scores = scores.masked_fill(causal_mask, float('-inf'))
        
        weights = F.softmax(scores, dim=-1)
        attention_out = torch.matmul(weights, V)
        X = X + self.Wo(attention_out)  # Residual
        
        # --- Feed-Forward with Pre-LN ---
        X_norm = self.ln2(X)
        FF = F.relu(self.W1(X_norm))  # ReLU is faster than GELU on P3
        X = X + self.W2(FF)  # Residual
        
        # --- Output ---
        X = self.ln_f(X)
        logits = self.lm_head(X)
        
        return logits
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """Generate text autoregressively"""
        self.eval()
        
        for _ in range(max_new_tokens):
            # Crop to context window
            idx_cond = idx if idx.size(1) <= SEQ_LEN else idx[:, -SEQ_LEN:]
            
            # Forward pass
            logits = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            
            # Optional top-k sampling
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            
            # Sample next token
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)
        
        return idx
    
    def count_parameters(self):
        """Count total parameters"""
        return sum(p.numel() for p in self.parameters())
    
    def memory_footprint(self):
        """Calculate memory requirements"""
        params = self.count_parameters()
        return {
            'parameters': params,
            'float32_mb': params * 4 / (1024 * 1024),
            'int8_mb': params / (1024 * 1024),
        }

# ----------------------------
# Create and analyze model
# ----------------------------
model = TinyTransformer()

print("=" * 70)
print("PENTIUM III OPTIMIZED TINYLLM")
print("=" * 70)
print(f"Configuration:")
print(f"  - Sequence length:  {SEQ_LEN}")
print(f"  - Vocabulary:       {VOCAB_SIZE} (ASCII)")
print(f"  - Embedding dim:    {EMBED_DIM}")
print(f"  - FF hidden dim:    {FF_DIM}")
print(f"  - Attention heads:  {NUM_HEADS}")
print(f"  - Layers:           {NUM_LAYERS}")
print("=" * 70)

mem = model.memory_footprint()
print(f"\nMemory Requirements:")
print(f"  - Total parameters: {mem['parameters']:,}")
print(f"  - Memory (float32): {mem['float32_mb']:.2f} MB")
print(f"  - Memory (int8):    {mem['int8_mb']:.2f} MB")
print(f"  - Fits in 128MB:    {'✓ YES' if mem['float32_mb'] < 100 else '✗ NO'}")
print("=" * 70)

print(f"\nParameter Breakdown:")
for name, param in model.named_parameters():
    print(f"  {name:30s} {param.shape} = {param.numel():>7,} params")

print("\n" + "=" * 70)
print("PENTIUM III PERFORMANCE ESTIMATE")
print("=" * 70)
flops_per_token = mem['parameters'] * 2  # Rough estimate
print(f"  - FLOPs per token:  ~{flops_per_token:,}")
print(f"  - P3 @ 500 MFLOPS:  ~{flops_per_token / 500_000:.0f} ms/token")
print(f"  - Generation speed: ~{1000 / (flops_per_token / 500_000):.1f} tokens/sec")
print(f"  - P3 @ 1 GFLOPS:    ~{flops_per_token / 1_000_000:.0f} ms/token")
print(f"  - Generation speed: ~{1000 / (flops_per_token / 1_000_000):.1f} tokens/sec")
print("=" * 70)

# Test inference
print("\nTesting inference...")
test_input = torch.randint(0, VOCAB_SIZE, (1, 8))
with torch.no_grad():
    output = model(test_input)
print(f"  Input shape:  {test_input.shape}")
print(f"  Output shape: {output.shape}")
print("  ✓ Forward pass successful")

# Test generation
print("\nTesting text generation...")
prompt = "Hi "
context = torch.tensor([[ord(c) for c in prompt]])
generated = model.generate(context, max_new_tokens=20, temperature=0.8)
generated_text = ''.join([chr(int(t)) if 32 <= int(t) < 127 else '?' for t in generated[0]])
print(f"  Prompt:    '{prompt}'")
print(f"  Generated: '{generated_text}'")
print("  ✓ Generation successful (untrained)")

print("\n" + "=" * 70)
print("NEXT STEPS")
print("=" * 70)
print("1. Train on your dataset (Shakespeare, code, etc.)")
print("2. Export weights for C implementation")
print("3. Quantize to int8 (4x memory reduction)")
print("4. Implement inference in C for Pentium III")
print("=" * 70)
print("\nModel ready for training!")