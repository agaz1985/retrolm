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
    
    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load data
    data_path = "./data/training_data.txt"
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found!")
        print("Please create training_data.txt with your text data.")
        return
    
    print("\n" + "="*70)
    print("LOADING DATA")
    print("="*70)
    data = load_text_data(data_path)
    
    data_size = len(data)
    print(f"Dataset size: {data_size:,} characters")
    
    if data_size < 50000:
        print(f"⚠️  WARNING: Dataset is very small ({data_size:,} chars)")
        print("   Recommended minimum: 50,000+ characters")
        print("   Current dataset will likely overfit")
    
    # Calculate actual steps per epoch
    num_sequences = data_size // config.model.seq_len
    steps_per_epoch = max(1, num_sequences // config.training.batch_size)
    total_steps = config.training.epochs * steps_per_epoch
    
    print(f"\nTraining Configuration:")
    print(f"  Sequences available: {num_sequences:,}")
    print(f"  Steps per epoch: {steps_per_epoch}")
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
    param_to_data_ratio = model.count_parameters() / data_size
    print(f"\nParameter/Data ratio: {param_to_data_ratio:.2f}")
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
    
    model = train_model(model, data, train_config, device)
    
    # Test generation
    print("\n" + "="*70)
    print("TESTING GENERATION")
    print("="*70)
    print()
    
    test_prompts = ["The ", "I ", "We ", "Love "]
    for prompt in test_prompts:
        generate_text(model, prompt, 50, device=device, temperature=0.8)
        print()
    
    # Export weights
    print("\n" + "="*70)
    print("EXPORTING WEIGHTS")
    print("="*70)
    export_weights(model, config.model)

    save_config_dict = {
        'model': config.model.__dict__,
        'training': config.training.__dict__,
    }
    
    # Save checkpoint
    checkpoint_folder = Path("./checkpoints")
    checkpoint_folder.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_folder / "model_checkpoint.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': save_config_dict,
    }, checkpoint_path)
    print(f"\n✓ Checkpoint saved to {checkpoint_path}")
    
    print("\n" + "="*70)
    print("ALL DONE! 🎉")
    print("="*70)
    print("\nNext steps:")
    print("1. Check training/validation loss curves")
    print("2. If overfitting (val >> train), add more data or reduce model size")
    print("3. Test the C implementation with: ./retro_inference")

if __name__ == "__main__":
    main()