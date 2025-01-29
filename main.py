import sys
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from src.gui.chat_interface import ChatInterface
from src.utils.logging_utils import setup_logging

ROOT_DIR = Path(__file__).parent
sys.path.append(str(ROOT_DIR))

def load_deepseek_model(model_dir: Path):
    """Load DeepSeek model and tokenizer"""
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    return model, tokenizer

def main():
    try:
        # Setup logging
        log_dir = ROOT_DIR / "logs"
        logger = setup_logging(log_dir)
        logger.info("Initializing application...")

        # Load DeepSeek model
        model_dir = ROOT_DIR / "models" / "deepseek"
        model, tokenizer = load_deepseek_model(model_dir)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Create generation function
        def generate_text(prompt, max_length=100, temperature=0.7):
            import time
            start_time = time.time()
            
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            input_tokens = len(inputs.input_ids[0])
            
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
            
            end_time = time.time()
            generation_time = end_time - start_time
            
            decoded_outputs = [tokenizer.decode(out, skip_special_tokens=True) for out in outputs]
            new_tokens = [len(out) - input_tokens for out in outputs]
            
            return {
                "texts": decoded_outputs,
                "metrics": {
                    "generation_time": generation_time,
                    "tokens_per_second": sum(new_tokens) / generation_time,
                    "input_tokens": input_tokens,
                    "new_tokens": new_tokens
                }
            }

        # Initialize GUI
        app = ChatInterface(
            model=model,
            tokenizer=tokenizer,
            device=device,
            generate_fn=generate_text
        )
        
        logger.info("GUI initialized successfully")
        app.run()

    except Exception as e:
        logger.error(f"Application error: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
