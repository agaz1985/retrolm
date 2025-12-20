import struct
import os
from pathlib import Path

def export_weights(model, model_config, output_dir='weights'):
    """Export model weights in binary format for C"""
    Path(output_dir).mkdir(exist_ok=True)
    model.eval()
    
    print(f"\n{'='*70}")
    print("EXPORTING WEIGHTS")
    print(f"{'='*70}")
    
    def save_matrix(tensor, name):
        """Save a 2D tensor as binary file"""
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)
        
        rows, cols = tensor.shape
        filepath = os.path.join(output_dir, f"{name}.bin")
        
        with open(filepath, 'wb') as f:
            f.write(struct.pack('II', rows, cols))
            f.write(tensor.detach().cpu().numpy().astype('float32').tobytes())
        
        print(f"  ✓ {name:30s} [{rows:4d} x {cols:4d}] = {rows*cols:6,} params")
        return rows * cols
    
    total_params = 0
    
    # Export all weights
    total_params += save_matrix(model.token_embed.weight, "token_embed")
    total_params += save_matrix(model.pos_embed, "pos_embed")
    total_params += save_matrix(model.Wq.weight, "Wq_weight")
    total_params += save_matrix(model.Wq.bias, "Wq_bias")
    total_params += save_matrix(model.Wk.weight, "Wk_weight")
    total_params += save_matrix(model.Wk.bias, "Wk_bias")
    total_params += save_matrix(model.Wv.weight, "Wv_weight")
    total_params += save_matrix(model.Wv.bias, "Wv_bias")
    total_params += save_matrix(model.Wo.weight, "Wo_weight")
    total_params += save_matrix(model.Wo.bias, "Wo_bias")
    total_params += save_matrix(model.W1.weight, "W1_weight")
    total_params += save_matrix(model.W1.bias, "W1_bias")
    total_params += save_matrix(model.W2.weight, "W2_weight")
    total_params += save_matrix(model.W2.bias, "W2_bias")
    total_params += save_matrix(model.lm_head.bias, "lm_head_bias")
    
    print(f"\n  Total parameters: {total_params:,}")
    print(f"  Size (float32): {total_params * 4 / (1024*1024):.2f} MB")
    print(f"{'='*70}\n")
    
    # Save config
    config_path = os.path.join(output_dir, "config.txt")
    with open(config_path, 'w') as f:
        f.write(f"SEQ_LEN={model_config.seq_len}\n")
        f.write(f"VOCAB_SIZE={model_config.vocab_size}\n")
        f.write(f"EMBED_DIM={model_config.embed_dim}\n")
        f.write(f"FF_DIM={model_config.ff_dim}\n")
    
    print(f"✓ Config saved to {config_path}")
    print(f"✓ All weights saved to {output_dir}/")