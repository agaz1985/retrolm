import torch
import os

def load_text_data(file_path=None, train_path=None, val_path=None, train_limit_mb=None):
    """Load and preprocess text data using byte-level encoding
    
    Args:
        file_path: Single file to load and split 90/10 (legacy mode)
        train_path: Separate training file (recommended)
        val_path: Separate validation file (recommended)
        train_limit_mb: Limit training data to first N megabytes (None = use all)
    
    Returns:
        tuple: (train_data, val_data) as torch tensors
    """
    # Mode 1: Separate train/val files (recommended for large datasets like TinyStories)
    if train_path and val_path:
        print(f"Loading separate train/val files...")
        
        with open(train_path, 'rb') as f:
            if train_limit_mb is not None:
                # Read only first N MB for faster training
                limit_bytes = train_limit_mb * 1024 * 1024
                train_bytes = f.read(limit_bytes)
                print(f"⚠️  Limited training data to {train_limit_mb} MB ({len(train_bytes):,} bytes)")
            else:
                train_bytes = f.read()
        train_data = torch.tensor(list(train_bytes), dtype=torch.long)
        
        with open(val_path, 'rb') as f:
            val_bytes = f.read()
        val_data = torch.tensor(list(val_bytes), dtype=torch.long)
        
        # Display info
        try:
            train_text = train_bytes[:100].decode('utf-8', errors='replace')
            val_text = val_bytes[:100].decode('utf-8', errors='replace')
        except:
            train_text = str(train_bytes[:100])
            val_text = str(val_bytes[:100])
        
        print(f"\nTrain set: {len(train_data):,} bytes")
        print(f"Val set: {len(val_data):,} bytes")
        print(f"Unique train chars: {len(set(train_bytes))}")
        print(f"Unique val chars: {len(set(val_bytes))}")
        print(f"\nTrain sample: {train_text}")
        print(f"Val sample: {val_text}")
        
        return train_data, val_data
    
    # Mode 2: Single file with 90/10 split (legacy)
    elif file_path:
        print(f"Loading single file with 90/10 split...")
        
        with open(file_path, 'rb') as f:
            raw_bytes = f.read()
        
        data = torch.tensor(list(raw_bytes), dtype=torch.long)
        
        # Split 90/10
        n = int(0.9 * len(data))
        train_data = data[:n]
        val_data = data[n:]
        
        try:
            text = raw_bytes[:100].decode('utf-8', errors='replace')
        except:
            text = str(raw_bytes[:100])
        
        print(f"\nTotal: {len(data):,} bytes")
        print(f"Train: {len(train_data):,} bytes")
        print(f"Val: {len(val_data):,} bytes")
        print(f"Unique chars: {len(set(raw_bytes))}")
        print(f"Sample: {text}")
        
        return train_data, val_data
    
    else:
        raise ValueError("Must provide either file_path OR both train_path and val_path")

def get_batch(train_data, val_data, batch_size, seq_len, split='train', device='cpu'):
    """Generate a batch of training data with proper consecutive sequences
    
    Args:
        train_data: Training data tensor
        val_data: Validation data tensor
        batch_size: Number of sequences per batch
        seq_len: Length of each sequence
        split: 'train' or 'val'
        device: Device to place tensors on
    """
    split_data = train_data if split == 'train' else val_data
    
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


def create_dataloader(train_data, val_data, batch_size, seq_len, split='train', device='cpu'):
    """
    Create a proper dataloader that iterates through data sequentially.
    This is better for small datasets as it ensures all data is seen each epoch.
    
    Args:
        train_data: Training data tensor
        val_data: Validation data tensor
        batch_size: Number of sequences per batch
        seq_len: Length of each sequence
        split: 'train' or 'val'
        device: Device to place tensors on
    """
    split_data = train_data if split == 'train' else val_data
    
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
    
    # Create batches (including remainder)
    num_full_batches = num_sequences // batch_size
    remainder = num_sequences % batch_size
    
    if num_full_batches == 0:
        # If we can't make a full batch, just use what we have
        return [(x.to(device), y.to(device))]
    
    batches = []
    for i in range(num_full_batches):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size
        batches.append((
            x[start_idx:end_idx].to(device),
            y[start_idx:end_idx].to(device)
        ))
    
    # Add remainder batch if it exists (important for small datasets)
    if remainder > 0:
        batches.append((
            x[num_full_batches * batch_size:].to(device),
            y[num_full_batches * batch_size:].to(device)
        ))
    
    return batches