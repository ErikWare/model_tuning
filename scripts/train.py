import os
import sys
from pathlib import Path
from typing import Optional, Union, Dict, Any
import json
from datetime import datetime

# ML/DL imports
import torch
from torch.utils.data import Dataset, DataLoader

# Hugging Face imports
from transformers import (
    GPT2LMHeadModel,     # Model architecture
    GPT2Tokenizer,       # Tokenizer
    Trainer,             # HF Trainer utility
    TrainingArguments,   # Training configuration
    DataCollatorForLanguageModeling  # Data collation for LM
)

# Hydra imports for config management
import hydra
from omegaconf import DictConfig

# Add project root to Python path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from src.utils.download_utils import ensure_directory, setup_logging

# SECTION: Global Configuration
ROOT_DIR = Path(__file__).resolve().parent.parent
logger = setup_logging()

# SECTION: Type Definitions
ModelType = Union[GPT2LMHeadModel, Any]  # Extend with other model types
TokenizerType = Union[GPT2Tokenizer, Any] # Extend with other tokenizers
DeviceType = Union[str, torch.device]

def resolve_path(cfg_path: str) -> Path:
    """Convert config path to absolute path relative to project root."""
    path = Path(cfg_path)
    if not path.is_absolute():
        path = ROOT_DIR / path
    return path

def setup_model_and_tokenizer(
    cfg: DictConfig
) -> tuple[ModelType, TokenizerType, DeviceType]:
    """
    Initialize model and tokenizer based on configuration.
    
    Args:
        cfg: Hydra configuration object containing model paths and settings
        
    Returns:
        tuple containing:
            - initialized model
            - initialized tokenizer
            - device (cuda/cpu)
    """
    # Load model and tokenizer from saved paths
    model = GPT2LMHeadModel.from_pretrained(cfg.model.model_path)
    tokenizer = GPT2Tokenizer.from_pretrained(cfg.model.tokenizer_path)

    

    # Assign eos_token as pad_token
    tokenizer.pad_token = tokenizer.eos_token

    # Update the model's padding token
    model.config.pad_token_id = tokenizer.eos_token_id
    
    # Set up device - allow manual override from config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if cfg.project.device != "auto":
        device = torch.device(cfg.project.device)
    
    model.to(device)
    return model, tokenizer, device

class MathDataset(Dataset):
    def __init__(self, data_path: str, tokenizer: TokenizerType):
        self.examples = []
        # Resolve the data path relative to project root
        resolved_path = resolve_path(data_path)
        if not resolved_path.exists():
            raise FileNotFoundError(f"Data file not found at: {resolved_path}")
            
        with open(resolved_path, 'r') as f:
            for line in f:
                example = json.loads(line.strip())
                # Format as instruction-following example
                text = f"Instruction: {example['prompt']}\nResponse: {example['completion']}"
                self.examples.append(text)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        text = self.examples[idx]
        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=512,
            padding='max_length',
            return_tensors='pt'
        )
        return {
            'input_ids': encoded['input_ids'].squeeze(),
            'attention_mask': encoded['attention_mask'].squeeze()
        }

def setup_training_args(cfg: DictConfig) -> TrainingArguments:
    """
    Initialize HuggingFace training arguments from config.
    
    Args:
        cfg: Hydra configuration containing training parameters
        
    Returns:
        TrainingArguments object
    """
    # Generate a unique run name using timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"training_run_{timestamp}"
    
    return TrainingArguments(
        output_dir=cfg.training.output_dir,
        run_name=run_name,  # Add distinct run name
        num_train_epochs=cfg.training.epochs,
        per_device_train_batch_size=cfg.training.batch_size,
        warmup_steps=cfg.training.warmup_steps,
        learning_rate=cfg.training.learning_rate,
        logging_dir=cfg.training.logging_dir,
        logging_steps=cfg.training.logging_steps,
        save_steps=cfg.training.save_steps,
        save_total_limit=cfg.training.save_total_limit
    )

@hydra.main(config_path="../config", config_name="config")
def train(cfg: DictConfig) -> None:
    """
    Main training function using Hydra for configuration.
    
    Training Pipeline:
    1. Load model and tokenizer
    2. Setup training arguments
    3. Prepare dataset
    4. Initialize trainer
    5. Train model
    6. Save results
    
    Args:
        cfg: Hydra configuration object
    """
    try:
        logger.info("Initializing training pipeline...")
        
        # Ensure data directory exists
        data_dir = ROOT_DIR / "data"
        data_dir.mkdir(exist_ok=True, parents=True)
        
        # SECTION: Model Setup
        model, tokenizer, device = setup_model_and_tokenizer(cfg)
        logger.info(f"Model initialized on {device}")
        
        # Log the resolved data path
        data_path = resolve_path(cfg.data.path)
        logger.info(f"Loading training data from: {data_path}")
        
        # SECTION: Dataset Setup
        dataset = MathDataset(cfg.data.path, tokenizer)
        logger.info(f"Loaded {len(dataset)} training examples")
        
        # SECTION: Training Setup
        training_args = setup_training_args(cfg)
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            data_collator=DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm=False
            )
        )
        
        # SECTION: Training Execution
        logger.info("Starting training...")
        trainer.train()
        
        # SECTION: Save Results
        model.save_pretrained(cfg.model.save_path)
        tokenizer.save_pretrained(cfg.model.save_path)
        logger.info(f"Model saved to {cfg.model.save_path}")
        
        logger.info("Training pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    train()
