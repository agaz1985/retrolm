import torch

def load_text_data(file_path):
    """Load and preprocess text data"""
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Filter to 7-bit ASCII (printable + newline)
    text = ''.join([c for c in text if 32 <= ord(c) < 127 or c == '\n'])
    
    # Convert to token IDs
    data = torch.tensor([ord(c) for c in text], dtype=torch.long)
    
    print(f"Loaded {len(data):,} characters")
    print(f"Unique characters: {len(set(text))}")
    print(f"Sample: {text[:100]}")
    
    return data

def get_batch(data, batch_size, seq_len, split='train', device='cpu'):
    """Generate a batch of training data with proper consecutive sequences"""
    # Split data 90/10 train/val
    n = int(0.9 * len(data))
    split_data = data[:n] if split == 'train' else data[n:]
    
    # Safety check
    if len(split_data) <= seq_len + 1:
        raise ValueError(
            f"{split.capitalize()} data too short! "
            f"Length: {len(split_data)}, but need seq_len+1={seq_len+1}. "
            "Need a larger text file or smaller seq_len."
        )
    
    # Random starting positions
    max_start = len(split_data) - seq_len - 1
    if max_start <= 0:
        raise ValueError(f"Not enough data for sequences. Need at least {seq_len + 1} chars")
    
    ix = torch.randint(0, max_start, (batch_size,))
    
    # Create input (x) and target (y) sequences
    # x: characters at positions i, i+1, ..., i+seq_len-1
    # y: characters at positions i+1, i+2, ..., i+seq_len (shifted by 1)
    x = torch.stack([split_data[i:i + seq_len] for i in ix])
    y = torch.stack([split_data[i + 1:i + seq_len + 1] for i in ix])
    
    return x.to(device), y.to(device)


def create_dataloader(data, batch_size, seq_len, split='train', device='cpu'):
    """
    Alternative: Create a proper dataloader that iterates through data sequentially.
    This is better for small datasets as it ensures all data is seen each epoch.
    """
    n = int(0.9 * len(data))
    split_data = data[:n] if split == 'train' else data[n:]
    
    # Calculate number of full sequences we can make
    num_sequences = (len(split_data) - 1) // seq_len
    
    if num_sequences == 0:
        raise ValueError(f"Data too short for even one sequence of length {seq_len}")
    
    # Truncate to fit evenly into sequences
    truncated_length = num_sequences * seq_len
    split_data = split_data[:truncated_length + 1]  # +1 for targets
    
    # Reshape into sequences
    x = split_data[:-1].view(num_sequences, seq_len)
    y = split_data[1:truncated_length + 1].view(num_sequences, seq_len)
    
    # Create batches
    num_batches = num_sequences // batch_size
    if num_batches == 0:
        # If we can't make a full batch, just use what we have
        return [(x.to(device), y.to(device))]
    
    batches = []
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size
        batches.append((
            x[start_idx:end_idx].to(device),
            y[start_idx:end_idx].to(device)
        ))
    
    return batches