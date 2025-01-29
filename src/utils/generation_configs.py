"""
Generation Configurations Module

This module provides pre-set parameter configurations for the `model.generate()` method
using Hugging Face's Transformers library. These configurations are designed to enhance
the quality and behavior of generated responses based on different requirements and use cases.

Usage:
    from utils.generation_configs import GenerationConfig

    selected_params = GenerationConfig.STANDARD_QUALITY
    response = model.generate(**inputs, **selected_params)
"""

from transformers import StoppingCriteriaList
from typing import Dict

class GenerationConfig:
    """
    A collection of preset configurations for text generation.
    Each configuration adjusts parameters to achieve different generation behaviors.
    """

    STANDARD_QUALITY: Dict[str, any] = {
        "max_new_tokens": 300,
        "temperature": 0.7,
        "top_k": 50,
        "top_p": 0.9,
        "repetition_penalty": 1.2,
        "no_repeat_ngram_size": 3,
        "num_beams": 5,
        "early_stopping": True,
        "eos_token_id": None,   # To be set dynamically based on tokenizer
        "min_length": 100,
        "length_penalty": 1.0
    }

    CREATIVE_DIVERSE: Dict[str, any] = {
        "max_new_tokens": 500,
        "temperature": 1.0,
        "top_k": 100,
        "top_p": 0.95,
        "repetition_penalty": 1.0,
        "no_repeat_ngram_size": 2,
        "num_beams": 3,
        "early_stopping": False,
        "eos_token_id": None,   # To be set dynamically based on tokenizer
        "min_length": 150,
        "length_penalty": 0.8
    }

    CONSERVATIVE_DETERMINISTIC: Dict[str, any] = {
        "max_new_tokens": 300,
        "temperature": 0.5,
        "top_k": 0,
        "top_p": 1.0,
        "repetition_penalty": 1.5,
        "no_repeat_ngram_size": 4,
        "num_beams": 1,  # Greedy decoding
        "early_stopping": True,
        "eos_token_id": None,   # To be set dynamically based on tokenizer
        "min_length": 100,
        "length_penalty": 1.0
    }

    SHORT_CONCISE: Dict[str, any] = {
        "max_new_tokens": 100,
        "temperature": 0.7,
        "top_k": 50,
        "top_p": 0.9,
        "repetition_penalty": 1.2,
        "no_repeat_ngram_size": 3,
        "num_beams": 3,
        "early_stopping": True,
        "eos_token_id": None,   # To be set dynamically based on tokenizer
        "min_length": 50,
        "length_penalty": 1.0
    }

    LONG_DETAILED: Dict[str, any] = {
        "max_new_tokens": 700,
        "temperature": 0.8,
        "top_k": 50,
        "top_p": 0.9,
        "repetition_penalty": 1.1,
        "no_repeat_ngram_size": 3,
        "num_beams": 5,
        "early_stopping": True,
        "eos_token_id": None,   # To be set dynamically based on tokenizer
        "min_length": 200,
        "length_penalty": 1.0
    }

    DIVERSE_BEAM_SEARCH: Dict[str, any] = {
        "max_new_tokens": 300,
        "temperature": 0.8,
        "top_k": 50,
        "top_p": 0.9,
        "repetition_penalty": 1.2,
        "no_repeat_ngram_size": 3,
        "num_beams": 5,
        "early_stopping": True,
        "num_return_sequences": 3,  # Generate multiple diverse responses
        "eos_token_id": None,      # To be set dynamically based on tokenizer
        "min_length": 100,
        "length_penalty": 1.0
    }

    SAFE_CONTROLLED: Dict[str, any] = {
        "max_new_tokens": 300,
        "temperature": 0.7,
        "top_k": 50,
        "top_p": 0.9,
        "repetition_penalty": 1.5,
        "no_repeat_ngram_size": 4,
        "num_beams": 5,
        "early_stopping": True,
        "eos_token_id": None,   # To be set dynamically based on tokenizer
        "min_length": 100,
        "length_penalty": 1.0
    }

    DIRECT_RESPONSE: Dict[str, any] = {
        "max_new_tokens": 150,
        "temperature": 0.5,
        "top_k": 50,
        "top_p": 0.9,
        "repetition_penalty": 1.2,
        "no_repeat_ngram_size": 3,
        "num_beams": 2,
        "early_stopping": True,
        "eos_token_id": None,   # To be set dynamically based on tokenizer
        "min_length": 20,
        "length_penalty": 1.0
    }

    @classmethod
    def get_config(cls, config_name: str, eos_token_id: int) -> Dict[str, any]:
        """
        Retrieve a generation parameter configuration by name and set the EOS token ID.

        Args:
            config_name (str): The name of the configuration.
            eos_token_id (int): The EOS token ID from the tokenizer.

        Returns:
            dict: The generation parameters with the EOS token ID set.
        """
        config = getattr(cls, config_name.upper(), None)
        if config is None:
            raise ValueError(f"Configuration '{config_name}' not found.")
        config = config.copy()  # To avoid mutating the class attribute
        config["eos_token_id"] = eos_token_id
        return config