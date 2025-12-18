import torch

def load_text_data(file_path):
    """Load and preprocess text data"""
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Filter to 7-bit ASCII
    text = ''.join([c for c in text if ord(c) < 128])
    
    # Convert to token IDs
    data = torch.tensor([ord(c) for c in text], dtype=torch.long)
    
    print(f"Loaded {len(data):,} characters")
    return data

def get_batch(data, batch_size, seq_len, split='train', device='cpu'):
    """Generate a batch of training data"""
    # Split data
    n = int(0.9 * len(data))
    data = data[:n] if split == 'train' else data[n:]
    
    # Random positions
    ix = torch.randint(len(data) - seq_len, (batch_size,))
    
    # Create sequences
    x = torch.stack([data[i:i+seq_len] for i in ix])
    y = torch.stack([data[i+1:i+seq_len+1] for i in ix])
    
    x, y = x.to(device), y.to(device)
    return x, y