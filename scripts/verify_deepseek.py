import os
import torch
from pathlib import Path
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.utils.logging_utils import setup_logging

# Initialize logger using centralized logging
logger = setup_logging()

def verify_model():
    """Verify the downloaded DeepSeek model"""
    try:
        # Get model directory
        model_dir = Path(__file__).resolve().parent.parent / "models" / "deepseek"
        
        if not model_dir.exists():
            raise FileNotFoundError(f"Model directory not found at {model_dir}")
        
        logger.info("Loading model and tokenizer...")
        logger.info(f"Model directory: {model_dir}")
        
        # List files in directory
        logger.info("Files in model directory:")
        for file in model_dir.glob("*"):
            logger.info(f"- {file.name}")
        
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        # Test generation
        test_prompt = """
        USER QUESTION: you are answering the question of a 5 year old, the question is: "what is the captial city of China"? 
        RESPONSE SAFTY CONSTRAINT: YOU WILL NOT USE ANY INAPPROPRIATE LANGUAGE OR CONTENT.
        MODEL RESPONSE: PROVIDE A FACTUAL ANSWER TO THE QUESTION.
        """
    
        logger.info(f"\nTesting with prompt: {test_prompt}")
        
        inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=300,  # Adjusted based on input length
            # max_length=512,  # Removed to rely on max_new_tokens
            temperature=0.7,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            eos_token_id=tokenizer.eos_token_id,
            min_length=40      # Reduced to avoid forcing too many tokens
        )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info("\nModel Response:")
        logger.info("-" * 50)
        logger.info(response)
        logger.info("-" * 50)
        
        logger.info("\nVerification complete! Model is working correctly.")
        return True
        
    except Exception as e:
        logger.error(f"Verification failed: {str(e)}")
        return False

if __name__ == "__main__":
    verify_model()
