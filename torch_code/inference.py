import torch

def generate_text(model, prompt="Hello", max_tokens=50, 
                 temperature=0.8, top_k=40, device='cpu'):
    """Generate text from a prompt"""
    model.eval()
    model = model.to(device)
    
    # Convert prompt to tokens
    context = torch.tensor([[ord(c) for c in prompt]], dtype=torch.long).to(device)
    
    # Generate
    generated = model.generate(context, max_new_tokens=max_tokens,
                              temperature=temperature, top_k=top_k)
    
    # Decode
    text = ''.join([chr(int(t)) if 32 <= int(t) < 127 else '?' 
                   for t in generated[0]])
    
    print(f"Prompt: '{prompt}'")
    print(f"Generated: '{text}'")
    print()
    
    return text