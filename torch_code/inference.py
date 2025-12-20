import torch

def generate_text(model, prompt="Hello", max_tokens=50, 
                 temperature=0.8, top_k=40, device='cpu'):
    """Generate text from a prompt using byte-level encoding"""
    model.eval()
    model = model.to(device)
    
    # Convert prompt to bytes (0-255 token IDs)
    prompt_bytes = prompt.encode('utf-8')
    context = torch.tensor(
        [list(prompt_bytes)], 
        dtype=torch.long
    ).to(device)
    
    # Generate
    generated = model.generate(context, max_new_tokens=max_tokens,
                              temperature=temperature, top_k=top_k)
    
    # Decode bytes back to text with error handling
    token_bytes = bytes([int(t) % 256 for t in generated[0]])
    try:
        text = token_bytes.decode('utf-8', errors='replace')
    except:
        # Fallback: decode as latin-1 (never fails)
        text = token_bytes.decode('latin-1')
    
    print(f"Prompt: '{prompt}'")
    print(f"Generated: '{text}'")
    print()
    
    return text