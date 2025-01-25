import torch
import time
from typing import Dict, Any

def generate_text(prompt: str, model, tokenizer, max_length: int = 100, 
                 temperature: float = 0.7, device: str = "cpu") -> Dict[str, Any]:
    """Generate text using the model with metrics."""
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_tokens = len(inputs['input_ids'][0])
    
    # Generate with timing
    start_time = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_return_sequences=1,
            temperature=temperature,
            pad_token_id=tokenizer.eos_token_id
        )
    end_time = time.time()
    
    # Calculate metrics
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generation_time = end_time - start_time
    new_tokens = [len(output) - input_tokens for output in outputs]
    tokens_per_second = sum(new_tokens) / generation_time if generation_time > 0 else 0
    
    return {
        "texts": [generated_text],
        "metrics": {
            "generation_time": generation_time,
            "tokens_per_second": tokens_per_second,
            "input_tokens": input_tokens,
            "new_tokens": new_tokens
        }
    }
