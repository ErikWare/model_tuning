import tkinter as tk
from tkinter import ttk, scrolledtext
import threading
from typing import Dict, Any, Callable
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from src.utils.voice_to_text import VoiceToText

class ModelTestWindow:
    """GUI window for testing language models."""
    
    def __init__(self, model: GPT2LMHeadModel, tokenizer: GPT2Tokenizer, device: str,
                 generate_fn: Callable, tts_engine):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.generate_fn = generate_fn
        self.voice_to_text = VoiceToText()  # Re-enable voice to text
        self.tts_engine = tts_engine
        
        # Create main window
        self.root = tk.Tk()
        self.root.title("GPT-2 Model Test")
        self.root.geometry("800x600")
        
        # Create and pack widgets
        self.create_widgets()
    
    def create_widgets(self):
        """Create and layout all GUI widgets."""
        # Style configuration
        style = ttk.Style()
        style.configure("Stats.TLabel", font=("Arial", 9))
        
        # Main container
        main_container = ttk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Prompt section
        prompt_frame = ttk.LabelFrame(main_container, text="Enter Prompt", padding="5")
        prompt_frame.pack(fill=tk.X, pady=5)
        
        self.prompt_entry = scrolledtext.ScrolledText(
            prompt_frame, height=3, wrap=tk.WORD, font=("Arial", 10)
        )
        self.prompt_entry.pack(fill=tk.X, pady=5)
        
        # Voice input button
        self.voice_button = ttk.Button(prompt_frame, text="🎤 Voice Input", command=self._handle_voice_input)
        self.voice_button.pack(side=tk.LEFT, pady=5)
        
        # Controls section
        controls_frame = ttk.Frame(main_container)
        controls_frame.pack(fill=tk.X, pady=5)
        
        # Left controls
        left_controls = ttk.Frame(controls_frame)
        left_controls.pack(side=tk.LEFT)
        
        ttk.Label(left_controls, text="Temperature:").pack(side=tk.LEFT)
        self.temp_var = tk.StringVar(value="0.7")
        self.temp_entry = ttk.Entry(
            left_controls, textvariable=self.temp_var, width=5
        )
        self.temp_entry.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(left_controls, text="Max Length:").pack(side=tk.LEFT, padx=5)
        self.length_var = tk.StringVar(value="100")
        self.length_entry = ttk.Entry(
            left_controls, textvariable=self.length_var, width=5
        )
        self.length_entry.pack(side=tk.LEFT)
        
        # Right controls
        right_controls = ttk.Frame(controls_frame)
        right_controls.pack(side=tk.RIGHT)
        
        clear_btn = ttk.Button(
            right_controls, text="Clear", command=self.clear_all
        )
        clear_btn.pack(side=tk.RIGHT, padx=5)
        
        generate_btn = ttk.Button(
            right_controls, text="Generate", command=self.generate_response
        )
        generate_btn.pack(side=tk.RIGHT)
        
        # Add TTS toggle button
        self.tts_button = ttk.Button(
            main_container,
            text="TTS: Off",
            command=self.toggle_tts
        )
        self.tts_button.pack(pady=5)
        
        # Metrics section
        self.metrics_frame = ttk.LabelFrame(main_container, text="Performance Metrics", padding="5")
        self.metrics_frame.pack(fill=tk.X, pady=5)
        
        metrics_grid = ttk.Frame(self.metrics_frame)
        metrics_grid.pack(fill=tk.X, padx=5)
        
        # Metrics labels
        self.metrics_labels = {}
        metrics = [
            ("Generation Time:", "0.00 s"),
            ("Tokens/Second:", "0 t/s"),
            ("Input Tokens:", "0"),
            ("Generated Tokens:", "0"),
        ]
        
        for i, (label, value) in enumerate(metrics):
            ttk.Label(metrics_grid, text=label, style="Stats.TLabel").grid(
                row=i//2, column=i%2*2, padx=5, pady=2, sticky="e"
            )
            self.metrics_labels[label] = ttk.Label(
                metrics_grid, text=value, style="Stats.TLabel"
            )
            self.metrics_labels[label].grid(
                row=i//2, column=i%2*2+1, padx=5, pady=2, sticky="w"
            )
        
        # Response section
        response_frame = ttk.LabelFrame(main_container, text="Generated Response", padding="5")
        response_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.response_text = scrolledtext.ScrolledText(
            response_frame, wrap=tk.WORD, height=15, font=("Arial", 10)
        )
        self.response_text.pack(fill=tk.BOTH, expand=True)
    
    def update_metrics(self, metrics: Dict[str, Any]):
        """Update the metrics display."""
        self.metrics_labels["Generation Time:"].config(
            text=f"{metrics['generation_time']:.2f} s"
        )
        self.metrics_labels["Tokens/Second:"].config(
            text=f"{metrics['tokens_per_second']:.1f} t/s"
        )
        self.metrics_labels["Input Tokens:"].config(
            text=str(metrics['input_tokens'])
        )
        self.metrics_labels["Generated Tokens:"].config(
            text=str(sum(metrics['new_tokens']))
        )
    
    def clear_all(self):
        """Clear all input and output fields."""
        self.prompt_entry.delete("1.0", tk.END)
        self.response_text.delete("1.0", tk.END)
        for label in self.metrics_labels.values():
            label.config(text="0")
    
    def generate_response(self):
        """Generate response with metrics."""
        prompt = self.prompt_entry.get("1.0", tk.END).strip()
        if not prompt:
            return
            
        try:
            temperature = float(self.temp_var.get())
            max_length = int(self.length_var.get())
            
            # Use the provided generate function
            result = self.generate_fn(
                prompt=prompt,
                model=self.model,
                tokenizer=self.tokenizer,
                max_length=max_length,
                temperature=temperature,
                device=self.device
            )
            
            # Update display
            self.response_text.delete("1.0", tk.END)
            for i, text in enumerate(result["texts"], 1):
                self.response_text.insert(tk.END, f"Generated text {i}:\n{text}\n\n")
            
            # Add TTS playback
            self.tts_engine.speak(result["texts"][0])
            
            # Update metrics
            self.update_metrics(result["metrics"])
                
        except Exception as e:
            self.response_text.delete("1.0", tk.END)
            self.response_text.insert(tk.END, f"Error: {str(e)}")
    
    def _handle_voice_input(self):
        """Handle voice input button click"""
        self.voice_button.state(['disabled'])
        self.root.update()
        
        def listen():
            text = self.voice_to_text.listen()
            if text:
                self.prompt_entry.delete('1.0', tk.END)
                self.prompt_entry.insert('1.0', text)
            self.voice_button.state(['!disabled'])
            
        # Run voice recognition in separate thread
        threading.Thread(target=listen, daemon=True).start()
    
    def toggle_tts(self):
        enabled = self.tts_engine.toggle()
        self.tts_button.config(text=f"TTS: {'On' if enabled else 'Off'}")
    
    def run(self):
        """Start the GUI event loop."""
        self.root.mainloop()
