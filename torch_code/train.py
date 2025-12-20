import torch
import torch.nn.functional as F

def train_model(model, train_data, val_data, config, device='cpu'):
    """Train the model with proper regularization and early stopping
    
    Args:
        model: The model to train
        train_data: Training data tensor
        val_data: Validation data tensor
        config: Training configuration dict
        device: Device to train on
    """
    model = model.to(device)
    
    # Use optimizer and scheduler from config (created in main.py)
    optimizer = config.get('optimizer')
    scheduler = config.get('scheduler')
    
    # If not provided, create them (fallback)
    if optimizer is None:
        print("⚠️  Creating optimizer (should be passed from config)")
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=config['learning_rate'],
            weight_decay=config.get('weight_decay', 0.1)
        )
    
    from data import get_batch, create_dataloader
    
    # Option: Use sequential dataloader for better training
    use_sequential = config.get('use_sequential_loader', False)
    
    if use_sequential:
        print("Using sequential dataloader (recommended for small datasets)")
        train_batches = create_dataloader(train_data, val_data, config['batch_size'], config['seq_len'], 'train', device)
        config['steps_per_epoch'] = len(train_batches)
        print(f"Created {len(train_batches)} batches per epoch")
    
    print(f"\n{'='*70}")
    print("TRAINING STARTED")
    print(f"{'='*70}")
    print(f"Device: {device}")
    if device == 'mps':
        print("Note: First batch on MPS may take 30-60 seconds to compile")
    print(f"Parameters: {model.count_parameters():,}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Learning rate: {config['learning_rate']}")
    print(f"Weight decay: {config.get('weight_decay', 0.0)}")
    print(f"Dropout: {config.get('dropout', 0.0)}")
    print(f"Gradient clip: {config.get('grad_clip', 1.0)}")
    print(f"Epochs: {config['epochs']}")
    print(f"Steps per epoch: {config['steps_per_epoch']}")
    print(f"{'='*70}\n")
    
    # Early stopping setup (more patient for larger datasets)
    best_val_loss = float('inf')
    patience = config.get('patience', 50)
    patience_counter = 0
    best_model_state = None
    
    global_step = 0
    
    # Test initial batch to warm up device
    print("\nWarming up device with test batch...")
    X_test, Y_test = get_batch(train_data, val_data, config['batch_size'], config['seq_len'], 'train', device)
    _ = model(X_test)
    print("✓ Device ready\n")
    import sys
    sys.stdout.flush()
    
    for epoch in range(config['epochs']):
        model.train()
        epoch_loss = 0
        print(f"\nEpoch {epoch+1}/{config['epochs']}")
        print("-" * 70)
        sys.stdout.flush()
        
        # Progress tracking
        log_interval = 100  # Log every 100 steps for consistent feedback
        
        for step in range(config['steps_per_epoch']):
            # Get batch
            if use_sequential and step < len(train_batches):
                X, Y = train_batches[step]
            else:
                X, Y = get_batch(train_data, val_data, config['batch_size'], config['seq_len'], 
                               'train', device)
            
            # Forward pass
            logits = model(X)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), Y.view(-1))
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            if config.get('grad_clip'):
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 
                    config['grad_clip']
                )
            
            optimizer.step()
            
            # Step learning rate scheduler
            if scheduler is not None:
                scheduler.step()
            
            epoch_loss += loss.item()
            global_step += 1
            
            # Progress indicator - show every 100 steps or first 10 steps
            if (step + 1) % log_interval == 0 or step < 10:
                progress = (step + 1) / config['steps_per_epoch'] * 100
                current_lr = optimizer.param_groups[0]['lr']
                avg_loss = epoch_loss / (step + 1)
                print(f"  Step {step+1:6d}/{config['steps_per_epoch']} ({progress:5.1f}%) | "
                      f"Loss: {loss.item():.4f} | Avg: {avg_loss:.4f} | LR: {current_lr:.2e}", 
                      end='\r')
                sys.stdout.flush()
        
        # Epoch statistics
        print()  # Clear progress line
        avg_train_loss = epoch_loss / config['steps_per_epoch']
        
        # Validation evaluation
        val_loss = evaluate_model(model, train_data, val_data, config, device)
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch {epoch+1:3d}/{config['epochs']} | "
              f"Train loss: {avg_train_loss:.4f} | "
              f"Val loss: {val_loss:.4f} | "
              f"LR: {current_lr:.2e}")
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f"  ✓ New best validation loss: {best_val_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n{'='*70}")
                print(f"Early stopping triggered at epoch {epoch+1}")
                print(f"Best validation loss: {best_val_loss:.4f}")
                print(f"{'='*70}\n")
                break
        
        # Warning if overfitting badly
        if val_loss - avg_train_loss > 0.5:
            print(f"  ⚠️  Large train/val gap: {val_loss - avg_train_loss:.4f}")
    
    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\n✓ Restored best model (val loss: {best_val_loss:.4f})")
    
    print(f"\n{'='*70}")
    print("TRAINING COMPLETED")
    print(f"{'='*70}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"{'='*70}\n")
    
    return model

@torch.no_grad()
def evaluate_model(model, train_data, val_data, config, device='cpu', eval_iters=None):
    """Evaluate model on validation set
    
    Args:
        model: The model to evaluate
        train_data: Training data tensor (not used, but kept for consistency)
        val_data: Validation data tensor
        config: Training configuration dict
        device: Device to evaluate on
        eval_iters: Number of iterations to evaluate (uses config value if None)
    """
    from data import get_batch
    
    model.eval()
    losses = []
    
    # Use config value if not specified
    if eval_iters is None:
        eval_iters = config.get('eval_iters', 50)
    
    # Reduce eval_iters if we don't have enough data
    actual_eval_iters = min(eval_iters, config.get('steps_per_epoch', 50))
    
    for _ in range(actual_eval_iters):
        try:
            X, Y = get_batch(train_data, val_data, config['batch_size'], config['seq_len'], 
                            'val', device)
            logits = model(X)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), Y.view(-1))
            losses.append(loss.item())
        except Exception as e:
            # Handle case where validation set is too small
            print(f"⚠️  Validation error: {e}")
            break
    
    if not losses:
        return float('inf')
    
    return sum(losses) / len(losses)