"""Configuration management for RetroLM"""
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional


@dataclass
class ModelConfig:
    """Model architecture configuration"""
    seq_len: int = 64
    vocab_size: int = 512
    embed_dim: int = 128
    ff_dim: int = 256
    num_layers: int = 1
    dropout: float = 0.1
    
    def validate(self):
        """Validate configuration values"""
        assert self.seq_len > 0, "seq_len must be positive"
        assert self.vocab_size > 0, "vocab_size must be positive"
        assert self.embed_dim > 0, "embed_dim must be positive"
        assert self.ff_dim > 0, "ff_dim must be positive"
        assert self.num_layers > 0, "num_layers must be positive"
        assert 0 <= self.dropout < 1, "dropout must be in [0, 1)"


@dataclass
class TrainingConfig:
    """Training hyperparameters configuration"""
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    epochs: int = 300
    warmup_steps: int = 500
    grad_clip: float = 1.0
    eval_interval: int = 100
    patience: int = 50
    use_sequential_loader: bool = False
    
    def validate(self):
        """Validate configuration values"""
        assert self.batch_size > 0, "batch_size must be positive"
        assert self.learning_rate > 0, "learning_rate must be positive"
        assert self.weight_decay >= 0, "weight_decay must be non-negative"
        assert self.epochs > 0, "epochs must be positive"
        assert self.warmup_steps >= 0, "warmup_steps must be non-negative"
        assert self.grad_clip > 0, "grad_clip must be positive"
        assert self.eval_interval > 0, "eval_interval must be positive"
        assert self.patience > 0, "patience must be positive"


@dataclass
class Config:
    """Complete configuration for RetroLM"""
    model: ModelConfig
    training: TrainingConfig
    
    def validate(self):
        """Validate all configurations"""
        self.model.validate()
        self.training.validate()
    
    def to_dict(self):
        """Convert config to dictionary"""
        return {
            'model': asdict(self.model),
            'training': asdict(self.training)
        }
    
    def save(self, filepath: str):
        """Save configuration to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"Configuration saved to {filepath}")


def load_config(filepath: Optional[str] = None) -> Config:
    """
    Load configuration from JSON file or use defaults
    
    Args:
        filepath: Path to JSON config file. If None, uses defaults.
        
    Returns:
        Config object with model and training configurations
    """
    if filepath is None or not Path(filepath).exists():
        if filepath is not None:
            print(f"Config file {filepath} not found, using defaults")
        config = Config(
            model=ModelConfig(),
            training=TrainingConfig()
        )
    else:
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Load model config
        model_data = data.get('model', {})
        model_config = ModelConfig(**model_data)
        
        # Load training config
        training_data = data.get('training', {})
        training_config = TrainingConfig(**training_data)
        
        config = Config(model=model_config, training=training_config)
        print(f"Configuration loaded from {filepath}")
    
    # Validate
    config.validate()
    
    return config


def print_config(config: Config):
    """Pretty print configuration"""
    print(f"\n{'='*70}")
    print("CONFIGURATION")
    print(f"{'='*70}")
    
    print("\nModel Configuration:")
    for key, value in asdict(config.model).items():
        print(f"  {key:20s}: {value}")
    
    print("\nTraining Configuration:")
    for key, value in asdict(config.training).items():
        print(f"  {key:20s}: {value}")
    
    print(f"{'='*70}\n")
