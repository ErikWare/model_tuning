import os
import torch
from pathlib import Path
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        USER QUESTION: I am 5 years old, tell me a story. 
        RESPONSE SAFTY CONSTRAINT: YOU WILL NOT USE ANY INAPPROPRIATE LANGUAGE OR CONTENT.
        """
    
        logger.info(f"\nTesting with prompt: {test_prompt}")
        
        inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            #max_length=2500,
            max_new_tokens=500,
            temperature=0.7,
            num_return_sequences=1
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
