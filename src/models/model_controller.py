import torch
import time
from pathlib import Path
from typing import Dict, Any, Union
import logging
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from src.utils.logging_utils import setup_logging

class ModelController:
    def __init__(self, model_dir: Union[str, Path], logger: logging.Logger):
        self.model_dir = Path(model_dir)
        self.logger = logger
        self.model, self.tokenizer, self.device = self._load_model_and_tokenizer()
        
    def _load_model_and_tokenizer(self):
        """Load the saved model and tokenizer from specified directory."""
        if not self.model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {self.model_dir}")
        
        self.logger.info(f"Loading model and tokenizer from {self.model_dir}")
        model = GPT2LMHeadModel.from_pretrained(self.model_dir)
        tokenizer = GPT2Tokenizer.from_pretrained(self.model_dir)
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        model.eval()
        
        return model, tokenizer, device
    
    def generate_text(
        self,
        prompt: str,
        max_length: int = 100,
        temperature: float = 0.7,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate text from a prompt with performance metrics."""
        try:
            # Tokenize input
            encoded = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(self.device)
            
            input_tokens = len(encoded['input_ids'][0])
            
            # Generate with timing
            start_time = time.time()
            outputs = self.model.generate(
                **encoded,
                max_length=max_length,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                num_return_sequences=1,
                return_dict_in_generate=True,
                output_scores=True
            )
            generation_time = time.time() - start_time
            
            # Process outputs
            generated_sequences = outputs.sequences
            generated_texts = [self.tokenizer.decode(seq, skip_special_tokens=True) 
                             for seq in generated_sequences]
            
            return {
                "texts": generated_texts,
                "metrics": {
                    "generation_time": generation_time,
                    "input_tokens": input_tokens,
                    "new_tokens": [len(seq) - input_tokens for seq in generated_sequences],
                    "tokens_per_second": len(generated_sequences[0]) / generation_time
                }
            }
        except Exception as e:
            self.logger.error(f"Generation error: {e}")
            raise
    
    def cleanup(self):
        """Cleanup resources when done"""
        if hasattr(self, 'model'):
            self.model.cpu()
            del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
