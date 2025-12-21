import torch
import os
import sys
from pathlib import Path
from config import load_config, print_config
from model import RetroLLMTransformer, create_optimizer, get_lr_scheduler
from data import load_text_data
from train import train_model
from export import export_weights
from inference import generate_text

def main():
    # Load configuration from JSON file or use defaults
    config_path = "./config.json"
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    
    config = load_config(config_path)
    print_config(config)
    
    # Setup device - prefer MPS for Apple Silicon
    if torch.backends.mps.is_available():
        device = 'mps'
        print("Using Apple Silicon GPU (MPS)")
    elif torch.cuda.is_available():
        device = 'cuda'
        print("Using NVIDIA GPU (CUDA)")
    else:
        device = 'cpu'
        print("Using CPU")
    print(f"Device: {device}")
    
    # Load data from config paths
    print("\n" + "="*70)
    print("LOADING DATA")
    print("="*70)
    
    train_data, val_data = load_text_data(
        train_path=config.data.train_path,
        val_path=config.data.val_path,
        train_limit_mb=config.data.train_limit_mb
    )
    
    train_size = len(train_data)
    val_size = len(val_data)
    total_size = train_size + val_size
    
    print(f"\nDataset sizes:")
    print(f"  Training: {train_size:,} bytes")
    print(f"  Validation: {val_size:,} bytes")
    print(f"  Total: {total_size:,} bytes")
    
    if train_size < 50000:
        print(f"⚠️  WARNING: Training set is very small ({train_size:,} bytes)")
        print("   Recommended minimum: 50,000+ bytes")
        print("   Current dataset will likely overfit")
    
    # Calculate actual steps per epoch (using TRAIN data only)
    num_sequences = train_size // config.model.seq_len
    steps_per_epoch = max(1, num_sequences // config.training.batch_size)
    total_steps = config.training.epochs * steps_per_epoch
    
    print(f"\nTraining Configuration:")
    print(f"  Train sequences: {num_sequences:,}")
    print(f"  Steps per epoch: {steps_per_epoch:,}")
    print(f"  Total training steps: {total_steps:,}")
    print(f"  Batch size: {config.training.batch_size}")
    print(f"  Learning rate: {config.training.learning_rate}")
    print(f"  Weight decay: {config.training.weight_decay}")
    print(f"  Dropout: {config.model.dropout}")
    
    # Create model
    print("\n" + "="*70)
    print("INITIALIZING MODEL")
    print("="*70)
    model = RetroLLMTransformer(config.model)
    print(f"Model Configuration:")
    print(f"  Embed Dim: {config.model.embed_dim}")
    print(f"  FF Dim: {config.model.ff_dim}")
    print(f"  Seq Length: {config.model.seq_len}")
    print(f"  Vocab Size: {config.model.vocab_size}")
    print(f"  Parameters: {model.count_parameters():,}")
    
    # Calculate parameter-to-data ratio
    param_to_data_ratio = model.count_parameters() / train_size
    print(f"\nParameter/Data ratio: {param_to_data_ratio:.4f}")
    if param_to_data_ratio > 0.1:
        print("⚠️  WARNING: Model has too many parameters for dataset size")
        print("   Consider reducing embed_dim and ff_dim in config.json")
    
    # Create optimizer with proper weight decay separation
    optimizer = create_optimizer(model, config.training)
    scheduler = get_lr_scheduler(optimizer, config.training, total_steps)
    
    # Prepare training config dict
    train_config = {
        'batch_size': config.training.batch_size,
        'learning_rate': config.training.learning_rate,
        'weight_decay': config.training.weight_decay,
        'epochs': config.training.epochs,
        'warmup_steps': config.training.warmup_steps,
        'grad_clip': config.training.grad_clip,
        'dropout': config.model.dropout,
        'steps_per_epoch': steps_per_epoch,
        'eval_interval': max(10, config.training.epochs // 10),  # Eval 10 times during training
        'seq_len': config.model.seq_len,
        'optimizer': optimizer,
        'scheduler': scheduler,
        'patience': config.training.patience,
        'use_sequential_loader': config.training.use_sequential_loader,
    }
    
    # Train with early stopping
    print("\n" + "="*70)
    print("TRAINING")
    print("="*70)
    print("Training will use early stopping if validation loss stops improving")
    print()
    
    model = train_model(model, train_data, val_data, train_config, device)
    
    # Test generation
    print("\n" + "="*70)
    print("TESTING GENERATION")
    print("="*70)
    print()
    
    for prompt in config.inference.test_prompts:
        generate_text(
            model, 
            prompt, 
            config.inference.max_tokens,
            device=device, 
            temperature=config.inference.temperature,
            top_k=config.inference.top_k
        )
        print()
    
    # Export weights
    print("\n" + "="*70)
    print("EXPORTING WEIGHTS")
    print("="*70)
    export_weights(model, config.model, config.output.weights_dir)

    save_config_dict = {
        'model': config.model.__dict__,
        'training': config.training.__dict__,
    }
    
    # Save checkpoint
    checkpoint_path = Path(config.output.checkpoint_dir) / "model_checkpoint.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': save_config_dict,
    }, checkpoint_path)
    print(f"\n✓ Checkpoint saved to {checkpoint_path}")
    
    print("\n" + "="*70)
    print("ALL DONE! 🎉")
    print("="*70)

if __name__ == "__main__":
    main()