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

    # Standard quality: Focused, concise responses with minimal extra content.
    STANDARD_QUALITY: Dict[str, any] = {
        "max_new_tokens": 250,         # Lower max tokens to avoid verbosity
        "temperature": 0.7,
        "top_k": 50,
        "top_p": 0.9,
        "repetition_penalty": 1.3,       # Slightly increased to penalize repetition
        "no_repeat_ngram_size": 4,       # Prevents extended digressions
        "num_beams": 5,
        "early_stopping": True,
        "eos_token_id": None,           # Set dynamically
        "min_length": 50,              # Lower minimum to discourage extra content
        "length_penalty": 1.0
    }

    # Creative diverse: Enables creativity while aiming to confine to the user request.
    CREATIVE_DIVERSE: Dict[str, any] = {
        "max_new_tokens": 350,
        "temperature": 1.1,            # Slightly higher for creativity
        "top_k": 80,
        "top_p": 0.95,
        "repetition_penalty": 1.0,
        "no_repeat_ngram_size": 3,
        "num_beams": 3,
        "early_stopping": False,       # Allows creative exploration but may risk extra content
        "eos_token_id": None,
        "min_length": 80,              # Ensures sufficient answer length without excess
        "length_penalty": 0.8
    }

    # Conservative deterministic: Focused and minimalistic answers.
    CONSERVATIVE_DETERMINISTIC: Dict[str, any] = {
        "max_new_tokens": 150,         # Shorter responses to avoid undesired additions
        "temperature": 0.4,            # Lower temperature for predictable output
        "top_k": 0,
        "top_p": 1.0,
        "repetition_penalty": 1.6,
        "no_repeat_ngram_size": 5,     # Stronger constraint on extraneous content
        "num_beams": 1,                # Greedy decoding
        "early_stopping": True,
        "eos_token_id": None,
        "min_length": 40,
        "length_penalty": 1.0
    }

    # Short concise: Tailored for very brief responses.
    SHORT_CONCISE: Dict[str, any] = {
        "max_new_tokens": 80,
        "temperature": 0.7,
        "top_k": 50,
        "top_p": 0.9,
        "repetition_penalty": 1.3,
        "no_repeat_ngram_size": 4,
        "num_beams": 3,
        "early_stopping": True,
        "eos_token_id": None,
        "min_length": 20,
        "length_penalty": 1.0
    }

    # Long detailed: For comprehensive answers, yet restrained to the requested topic.
    LONG_DETAILED: Dict[str, any] = {
        "max_new_tokens": 700,
        "temperature": 0.8,
        "top_k": 50,
        "top_p": 0.9,
        "repetition_penalty": 1.2,
        "no_repeat_ngram_size": 4,     # Encourages coherence over lengthy output
        "num_beams": 5,
        "early_stopping": True,
        "eos_token_id": None,
        "min_length": 150,            # Ensures detail without veering off-topic
        "length_penalty": 1.0
    }

    # Diverse beam search: Modified to return only one sequence to prevent extra unrelated content.
    DIVERSE_BEAM_SEARCH: Dict[str, any] = {
        "max_new_tokens": 250,
        "temperature": 0.8,
        "top_k": 50,
        "top_p": 0.9,
        "repetition_penalty": 1.2,
        "no_repeat_ngram_size": 4,
        "num_beams": 5,
        "early_stopping": True,
        "num_return_sequences": 1,     # Return only one response to avoid extra content
        "eos_token_id": None,
        "min_length": 50,
        "length_penalty": 1.0
    }

    # Safe controlled: Emphasis on response safety and adherence to the query.
    SAFE_CONTROLLED: Dict[str, any] = {
        "max_new_tokens": 250,
        "temperature": 0.7,
        "top_k": 50,
        "top_p": 0.9,
        "repetition_penalty": 1.8,       # Higher penalty to discourage deviation
        "no_repeat_ngram_size": 5,       # Limits digressions
        "num_beams": 5,
        "early_stopping": True,
        "eos_token_id": None,
        "min_length": 50,
        "length_penalty": 1.0
    }

    # Direct response: For precise answers without any additional elaboration.
    DIRECT_RESPONSE: Dict[str, any] = {
        "max_new_tokens": 100,
        "temperature": 0.3,            # Very low temperature for strict adherence
        "top_k": 50,
        "top_p": 0.9,
        "repetition_penalty": 1.3,
        "no_repeat_ngram_size": 4,     # Prevents extra or tangential content
        "num_beams": 2,
        "early_stopping": True,
        "eos_token_id": None,
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