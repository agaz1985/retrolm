import torch
import os
from pathlib import Path

from model import RetroLLMTransformer, SEQ_LEN, VOCAB_SIZE, EMBED_DIM, FF_DIM
from data import load_text_data
from train import train_model
from export import export_weights
from inference import generate_text

# Training configuration
CONFIG = {
    'batch_size': 256,
    'lr': 5e-4,
    'epochs': 1000,
    'steps_per_epoch': 500,
    'eval_interval': 100,
    'seq_len': SEQ_LEN,
}

def main():
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
    
    if len(data) < 1000:
        print("Warning: Dataset is very small. Consider adding more data.")
    
    # Create model
    print("\n" + "="*70)
    print("INITIALIZING MODEL")
    print("="*70)
    model = RetroLLMTransformer()
    print(f"Parameters: {model.count_parameters():,}")
    
    # Train
    model = train_model(model, data, CONFIG, device)
    
    # Test generation
    print("\n" + "="*70)
    print("TESTING GENERATION")
    print("="*70)
    print()
    generate_text(model, "The ", 50, device=device)
    generate_text(model, "I ", 50, device=device)
    generate_text(model, "A ", 50, device=device)
    
    # Export weights
    export_weights(model)
    
    # Save checkpoint
    checkpoint_folder = Path("./checkpoints")
    checkpoint_folder.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_folder / "model_checkpoint.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': CONFIG,
    }, checkpoint_path)
    print(f"\n✓ Checkpoint saved to {checkpoint_path}")
    
    print("\n" + "="*70)
    print("ALL DONE! 🎉")
    print("="*70)

if __name__ == "__main__":
    main()