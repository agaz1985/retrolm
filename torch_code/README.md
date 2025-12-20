# RetroLM Training Code

PyTorch training code for the RetroLM transformer model.

## Configuration

All model and training parameters are now configured via `config.json`. You can customize:

### Model Parameters
- `seq_len`: Sequence length (default: 64)
- `vocab_size`: Vocabulary size (default: 512)
- `embed_dim`: Embedding dimension (default: 128)
- `ff_dim`: Feed-forward dimension (default: 256)
- `num_layers`: Number of transformer layers (default: 1)
- `dropout`: Dropout rate (default: 0.1)

### Training Parameters
- `batch_size`: Batch size (default: 128)
- `learning_rate`: Learning rate (default: 0.001)
- `weight_decay`: Weight decay for AdamW (default: 0.01)
- `epochs`: Number of training epochs (default: 300)
- `warmup_steps`: LR warmup steps (default: 500)
- `grad_clip`: Gradient clipping value (default: 1.0)
- `eval_interval`: Evaluation interval (default: 100)
- `patience`: Early stopping patience (default: 50)
- `use_sequential_loader`: Use sequential data loader (default: false)

## Usage

### Training with default config
```bash
python main.py
```

### Training with custom config
```bash
python main.py path/to/custom_config.json
```

### Example Custom Config
Create a custom JSON file with your parameters:
```json
{
  "model": {
    "seq_len": 128,
    "embed_dim": 256,
    "ff_dim": 512
  },
  "training": {
    "batch_size": 64,
    "learning_rate": 0.0005,
    "epochs": 500
  }
}
```

## Files

- `config.py` - Configuration dataclasses and JSON loading
- `config.json` - Default configuration file
- `model.py` - Transformer model architecture
- `train.py` - Training loop with early stopping
- `data.py` - Data loading and batching
- `export.py` - Export weights for C inference
- `inference.py` - Text generation
- `main.py` - Main training script