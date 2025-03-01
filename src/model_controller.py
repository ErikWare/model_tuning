import logging
import torch
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from src.utils.logging_utils import setup_logging
from src.utils.generation_configs import GenerationConfig

logger = setup_logging()

class ModelController:
    def __init__(
        self,
        model_path: Path,
        device: torch.device,
        model_kwargs: Optional[Dict] = None
    ):
        """
        Generic model controller that handles model loading and text generation.
        
        Args:
            model_path: Path to the model directory
            device: torch device (cuda/mps/cpu)
            model_kwargs: Optional model configuration overrides
        """
        self.model_path = model_path
        self.device = device
        self.model_kwargs = self._get_default_kwargs(device)
        
        # Override defaults with any provided kwargs
        if model_kwargs:
            self.model_kwargs.update(model_kwargs)
            
        logger.info(f"Initializing ModelController for model at: {model_path}")
        logger.info(f"Using device: {device}")
        logger.debug(f"Model configuration: {self.model_kwargs}")
        
        self._load_model()
    
    def _get_default_kwargs(self, device: torch.device) -> Dict:
        """Get default model configuration based on device."""
        kwargs = {
            "trust_remote_code": True, # watch out for this!
            "low_cpu_mem_usage": True
        }
        
        # Device-specific configurations
        if device.type == "mps":
            kwargs.update({
                "torch_dtype": torch.float32,
                "device_map": None
            })
        else:
            kwargs.update({
                "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
                "device_map": "auto"
            })
            
        return kwargs
    
    def _load_model(self):
        """Load the model and tokenizer."""
        try:
            logger.info("Loading tokenizer offline...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                use_fast=True,
                trust_remote_code=True,
                local_files_only=True
            )
            
            # Load configuration and ensure model_type exists
            config = AutoConfig.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                local_files_only=True
            )
            if not getattr(config, "model_type", None):
                logger.warning("config.json is missing 'model_type'. Setting default 'gpt2'.")
                config.__dict__["model_type"] = "gpt2"
            
            logger.info("Loading model securely using AutoModelForCausalLM.from_pretrained()...")
            # Use secure parameters: weights_only=True, local_files_only=True, low_cpu_mem_usage, etc.
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                **self.model_kwargs,
                local_files_only=True,
                weights_only=True  # ensure only model weights are loaded
            )
            
            # Move model to device if using MPS
            if self.device.type == "mps":
                logger.info("Moving model to MPS device")
                self.model = self.model.to(self.device)
            
            logger.info("Model and tokenizer loaded successfully.")
            
            # Get model info
            self.model_name = getattr(self.model.config, '_name_or_path', 'Unknown Model')
            self.model_params = sum(p.numel() for p in self.model.parameters()) / 1_000_000
            
            logger.info(f"Model loaded: {self.model_name} ({self.model_params:.1f}M parameters)")
            
        except Exception as e:
            logger.error("Failed to load model or tokenizer", exc_info=True)
            raise

    # Removed _load_model_weights() and _build_model() to avoid confusion.

    def generate(
        self,
        prompt: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate text with performance metrics."""
        try:
            import time
            start_time = time.time()
            
            # Process input
            inputs = self.tokenizer(prompt, return_tensors="pt")
            if self.device.type == "mps":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            input_tokens = len(inputs['input_ids'][0])
            
            # Generate text
            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    **kwargs
                )
            
            # Move outputs to CPU if needed
            if self.device.type == "mps":
                outputs = outputs.cpu()
            
            # Process results
            end_time = time.time()
            generation_time = end_time - start_time
            
            decoded_outputs = [self.tokenizer.decode(out, skip_special_tokens=True) 
                             for out in outputs]
            new_tokens = [len(out) - input_tokens for out in outputs]
            
            metrics = {
                "generation_time": generation_time,
                "tokens_per_second": sum(new_tokens) / generation_time,
                "input_tokens": input_tokens,
                "new_tokens": new_tokens
            }
            
            logger.debug(f"Generation metrics: {metrics}")
            return {
                "texts": decoded_outputs,
                "metrics": metrics
            }
            
        except Exception as e:
            logger.error("Text generation failed", exc_info=True)
            raise
            
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            "name": self.model_name,
            "parameters": self.model_params,
            "device": self.device,
            "config": self.model_kwargs
        }

