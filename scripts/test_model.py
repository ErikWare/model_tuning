import os
import sys
import torch
from pathlib import Path
import time
from typing import Dict, Any

# Add project root to Python path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

# Hugging Face imports
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Local imports
from src.utils.download_utils import setup_logging
from src.gui.model_test_window import ModelTestWindow

# Initialize logger and paths
logger = setup_logging()
MODEL_DIR = ROOT_DIR / "models" / "gpt2"

def load_model_and_tokenizer():
    """Load the saved model and tokenizer."""
    model_path = MODEL_DIR / "model"
    tokenizer_path = MODEL_DIR / "tokenizer"
    
    logger.info("Loading model and tokenizer...")
    model = GPT2LMHeadModel.from_pretrained(model_path)
    tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)
    
    # Move to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()  # Set to evaluation mode
    
    return model, tokenizer, device

def generate_text(
    prompt: str,
    model: GPT2LMHeadModel,
    tokenizer: GPT2Tokenizer,
    max_length: int = 100,
    temperature: float = 0.7,
    top_p: float = 0.9,
    num_return_sequences: int = 1,
    device: str = "cpu"
) -> Dict[str, Any]:
    """Generate text from a prompt with performance metrics."""
    metrics = {}
    
    # Tokenize and measure input
    start_time = time.time()
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
    input_tokens = len(inputs[0])
    
    # Generate
    generation_start = time.time()
    outputs = model.generate(
        inputs,
        max_length=max_length,
        temperature=temperature,
        top_p=top_p,
        num_return_sequences=num_return_sequences,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        return_dict_in_generate=True,
        output_scores=True
    )
    generation_end = time.time()
    
    # Calculate metrics
    generated_sequences = outputs.sequences
    output_tokens = [len(seq) for seq in generated_sequences]
    new_tokens = [ot - input_tokens for ot in output_tokens]
    
    generation_time = generation_end - generation_start
    tokens_per_second = sum(new_tokens) / generation_time if generation_time > 0 else 0
    
    # Decode outputs
    generated_texts = [tokenizer.decode(seq, skip_special_tokens=True) 
                      for seq in generated_sequences]
    
    return {
        "texts": generated_texts,
        "metrics": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "new_tokens": new_tokens,
            "generation_time": generation_time,
            "tokens_per_second": tokens_per_second,
        }
    }

def main():
    # Load model
    model, tokenizer, device = load_model_and_tokenizer()
    logger.info(f"Model loaded on {device}")
    
    # Create and run GUI with generate_text function
    app = ModelTestWindow(
        model=model,
        tokenizer=tokenizer,
        device=device,
        generate_fn=generate_text
    )
    app.run()

if __name__ == "__main__":
    main()
