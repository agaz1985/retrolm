"""Configuration management for RetroLM"""
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, List


@dataclass
class DataConfig:
    """Data paths configuration"""
    train_path: str = "./data/TinyStories-train.txt"
    val_path: str = "./data/TinyStories-valid.txt"
    train_limit_mb: Optional[int] = None  # Limit training data size in MB (None = use all)
    
    def validate(self):
        """Validate data paths exist"""
        if not Path(self.train_path).exists():
            raise FileNotFoundError(f"Training data not found: {self.train_path}")
        if not Path(self.val_path).exists():
            raise FileNotFoundError(f"Validation data not found: {self.val_path}")
        if self.train_limit_mb is not None and self.train_limit_mb <= 0:
            raise ValueError(f"train_limit_mb must be positive, got {self.train_limit_mb}")


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
    eval_iters: int = 50  # Number of iterations for evaluation
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
        assert self.eval_iters > 0, "eval_iters must be positive"
        assert self.patience > 0, "patience must be positive"


@dataclass
class OutputConfig:
    """Output directories configuration"""
    weights_dir: str = "./weights"
    checkpoint_dir: str = "./checkpoints"
    
    def validate(self):
        """Create output directories if they don't exist"""
        Path(self.weights_dir).mkdir(parents=True, exist_ok=True)
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)


@dataclass
class InferenceConfig:
    """Inference parameters configuration"""
    test_prompts: List[str] = None
    max_tokens: int = 50
    temperature: float = 0.8
    top_k: int = 40
    
    def __post_init__(self):
        if self.test_prompts is None:
            self.test_prompts = ["The ", "I ", "We ", "Love "]
    
    def validate(self):
        """Validate inference parameters"""
        assert self.max_tokens > 0, "max_tokens must be positive"
        assert self.temperature > 0, "temperature must be positive"
        assert self.top_k > 0, "top_k must be positive"


@dataclass
class Config:
    """Complete configuration for RetroLM"""
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    output: OutputConfig
    inference: InferenceConfig
    
    def validate(self):
        """Validate all configurations"""
        self.data.validate()
        self.model.validate()
        self.training.validate()
        self.output.validate()
        self.inference.validate()
    
    def to_dict(self):
        """Convert config to dictionary"""
        return {
            'data': asdict(self.data),
            'model': asdict(self.model),
            'training': asdict(self.training),
            'output': asdict(self.output),
            'inference': asdict(self.inference)
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
        Config object with all configurations
    """
    if filepath is None or not Path(filepath).exists():
        if filepath is not None:
            print(f"Config file {filepath} not found, using defaults")
        config = Config(
            data=DataConfig(),
            model=ModelConfig(),
            training=TrainingConfig(),
            output=OutputConfig(),
            inference=InferenceConfig()
        )
    else:
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Load all config sections
        data_config = DataConfig(**data.get('data', {}))
        model_config = ModelConfig(**data.get('model', {}))
        training_config = TrainingConfig(**data.get('training', {}))
        output_config = OutputConfig(**data.get('output', {}))
        inference_config = InferenceConfig(**data.get('inference', {}))
        
        config = Config(
            data=data_config,
            model=model_config,
            training=training_config,
            output=output_config,
            inference=inference_config
        )
        print(f"Configuration loaded from {filepath}")
    
    # Validate
    config.validate()
    
    return config


def print_config(config: Config):
    """Pretty print configuration"""
    print(f"\n{'='*70}")
    print("CONFIGURATION")
    print(f"{'='*70}")
    
    print("\nData Configuration:")
    for key, value in asdict(config.data).items():
        print(f"  {key:20s}: {value}")
    
    print("\nModel Configuration:")
    for key, value in asdict(config.model).items():
        print(f"  {key:20s}: {value}")
    
    print("\nTraining Configuration:")
    for key, value in asdict(config.training).items():
        print(f"  {key:20s}: {value}")
    
    print("\nOutput Configuration:")
    for key, value in asdict(config.output).items():
        print(f"  {key:20s}: {value}")
    
    print("\nInference Configuration:")
    for key, value in asdict(config.inference).items():
        if key == 'test_prompts':
            print(f"  {key:20s}: {value[:3]}..." if len(value) > 3 else f"  {key:20s}: {value}")
        else:
            print(f"  {key:20s}: {value}")
    
    print(f"{'='*70}\n")
