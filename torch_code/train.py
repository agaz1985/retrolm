import torch
import torch.nn.functional as F

def train_model(model, data, config, device='cpu'):
    """Train the model"""
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'])
    
    from data import get_batch
    
    print(f"\n{'='*70}")
    print("TRAINING STARTED")
    print(f"{'='*70}")
    print(f"Device: {device}")
    print(f"Parameters: {model.count_parameters():,}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Learning rate: {config['lr']}")
    print(f"Epochs: {config['epochs']}")
    print(f"{'='*70}\n")
    
    for epoch in range(config['epochs']):
        model.train()
        epoch_loss = 0
        
        for step in range(config['steps_per_epoch']):
            # Get batch
            X, Y = get_batch(data, config['batch_size'], config['seq_len'], 
                           'train', device)
            
            # Forward pass
            logits = model(X)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), Y.view(-1))
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            # Logging
            if step % config['eval_interval'] == 0:
                val_loss = evaluate_model(model, data, config, device)
                print(f"Epoch {epoch+1}/{config['epochs']} | "
                      f"Step {step}/{config['steps_per_epoch']} | "
                      f"Train loss: {loss.item():.4f} | Val loss: {val_loss:.4f}")
                model.train()
        
        avg_loss = epoch_loss / config['steps_per_epoch']
        print(f"\nEpoch {epoch+1} average loss: {avg_loss:.4f}\n")
    
    print(f"{'='*70}")
    print("TRAINING COMPLETED")
    print(f"{'='*70}\n")
    
    return model

@torch.no_grad()
def evaluate_model(model, data, config, device='cpu', eval_iters=50):
    """Evaluate model on validation set"""
    from data import get_batch
    
    model.eval()
    losses = []
    
    for _ in range(eval_iters):
        X, Y = get_batch(data, config['batch_size'], config['seq_len'], 
                        'val', device)
        logits = model(X)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), Y.view(-1))
        losses.append(loss.item())
    
    return sum(losses) / len(losses)