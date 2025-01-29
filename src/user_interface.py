import os
os.environ['TK_SILENCE_DEPRECATION'] = '1'

import tkinter as tk
from tkinter import scrolledtext
import logging
from src.utils.logging_utils import setup_logging
import threading

logger = setup_logging()

class ChatInterface:
    def __init__(self, model=None, tokenizer=None, device='cpu', generate_fn=None, tts_engine=None):
        # Save the generation function
        self.generate_fn = generate_fn
        
        # Create main window
        self.root = tk.Tk()
        self.root.title("AI Chat Interface")
        self.root.geometry("800x900")  # Increased height for new elements
        self.root.configure(bg='white')
        
        # Create main container
        main_frame = tk.Frame(self.root, bg='white')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Model Status Frame
        status_frame = tk.LabelFrame(main_frame, text="Model Status", bg='white')
        status_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Get model info
        model_name = getattr(model.config, '_name_or_path', 'Unknown Model')
        model_params = sum(p.numel() for p in model.parameters()) / 1_000_000
        
        # Display model information
        tk.Label(
            status_frame,
            text=f"Loaded Model: {model_name}",
            bg='white',
            font=('Arial', 10, 'bold')
        ).pack(anchor='w', pady=2)
        
        tk.Label(
            status_frame,
            text=f"Parameters: {model_params:.1f}M",
            bg='white'
        ).pack(anchor='w', pady=2)
        
        tk.Label(
            status_frame,
            text=f"Device: {device}",
            bg='white'
        ).pack(anchor='w', pady=2)
        
        # Controls Frame with better organization
        controls_frame = tk.LabelFrame(main_frame, text="Generation Controls", bg='white')
        controls_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Temperature control
        temp_frame = tk.Frame(controls_frame, bg='white')
        temp_frame.pack(side=tk.LEFT, padx=10, pady=5)
        tk.Label(temp_frame, text="Temperature:", bg='white').pack(side=tk.LEFT)
        self.temp_var = tk.StringVar(value="0.7")
        tk.Entry(temp_frame, textvariable=self.temp_var, width=5).pack(side=tk.LEFT, padx=5)
        
        # Max length control
        length_frame = tk.Frame(controls_frame, bg='white')
        length_frame.pack(side=tk.LEFT, padx=10, pady=5)
        tk.Label(length_frame, text="Max Length:", bg='white').pack(side=tk.LEFT)
        self.length_var = tk.StringVar(value="100")
        tk.Entry(length_frame, textvariable=self.length_var, width=5).pack(side=tk.LEFT, padx=5)
        
        # Metrics Frame with grid layout
        metrics_frame = tk.LabelFrame(main_frame, text="Generation Metrics", bg='white')
        metrics_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.metrics_labels = {}
        metrics_info = [
            ("Generation Time", "0.00s"),
            ("Tokens/Second", "0 t/s"),
            ("Input Tokens", "0"),
            ("Output Tokens", "0")
        ]
        
        for i, (metric, default) in enumerate(metrics_info):
            row = i // 2
            col = (i % 2) * 2
            
            tk.Label(
                metrics_frame,
                text=f"{metric}:",
                bg='white',
                font=('Arial', 10)
            ).grid(row=row, column=col, padx=5, pady=5, sticky='e')
            
            self.metrics_labels[metric] = tk.Label(
                metrics_frame,
                text=default,
                bg='white',
                font=('Arial', 10, 'bold')
            )
            self.metrics_labels[metric].grid(row=row, column=col+1, padx=5, pady=5, sticky='w')
        
        # Input area
        tk.Label(main_frame, text="Enter your prompt:", bg='white', font=('Arial', 10)).pack(anchor='w')
        self.input_text = scrolledtext.ScrolledText(
            main_frame,
            height=6,
            font=('Arial', 11),
            wrap=tk.WORD
        )
        self.input_text.pack(fill=tk.X, pady=(0, 10))
        
        # Generate button
        self.generate_btn = tk.Button(
            main_frame,
            text="Generate Response",
            command=self.generate_response,
            font=('Arial', 11),
            pady=5
        )
        self.generate_btn.pack(pady=(0, 20))
        
        # Output area
        tk.Label(main_frame, text="Response:", bg='white', font=('Arial', 10)).pack(anchor='w')
        self.output_text = scrolledtext.ScrolledText(
            main_frame,
            height=12,
            font=('Arial', 11),
            wrap=tk.WORD
        )
        self.output_text.pack(fill=tk.BOTH, expand=True)
        
        # Add keyboard shortcut
        self.root.bind('<Control-Return>', lambda e: self.generate_response())
    
    def update_metrics(self, metrics):
        """Update metrics display with animation"""
        self.metrics_labels["Generation Time"].config(text=f"{metrics['generation_time']:.2f}s")
        self.metrics_labels["Tokens/Second"].config(text=f"{metrics['tokens_per_second']:.1f} t/s")
        self.metrics_labels["Input Tokens"].config(text=str(metrics['input_tokens']))
        self.metrics_labels["Output Tokens"].config(text=str(sum(metrics['new_tokens'])))
        
        # Highlight updated metrics briefly
        for label in self.metrics_labels.values():
            label.config(foreground='blue')
            self.root.after(1000, lambda l=label: l.config(foreground='black'))
    
    def generate_response(self):
        """Generate response based on input text asynchronously."""
        thread = threading.Thread(target=self._generate_response_thread)
        thread.start()
    
    def _generate_response_thread(self):
        """Thread target for generating response."""
        try:
            self.generate_btn.config(state='disabled')
            prompt = self.input_text.get("1.0", tk.END).strip()
            
            if not prompt:
                return
                
            # Get parameters from controls
            temperature = float(self.temp_var.get())
            max_length = int(self.length_var.get())
            
            result = self.generate_fn(
                prompt=prompt,
                max_length=max_length,
                temperature=temperature
            )
            
            # Update metrics with animation
            self.update_metrics(result["metrics"])
            
            # Display response
            self.output_text.delete("1.0", tk.END)
            self.output_text.insert(tk.END, result["texts"][0])
            
        except Exception as e:
            logger.error(f"Generation error: {str(e)}")
            self.output_text.delete("1.0", tk.END)
            self.output_text.insert(tk.END, f"Error: {str(e)}")
        finally:
            self.generate_btn.config(state='normal')
    
    def run(self):
        """Start the GUI."""
        self.root.mainloop()
