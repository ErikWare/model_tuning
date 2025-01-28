import os
os.environ['TK_SILENCE_DEPRECATION'] = '1'

import tkinter as tk
from tkinter import scrolledtext
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChatInterface:
    def __init__(self, model=None, tokenizer=None, device='cpu', generate_fn=None, tts_engine=None):
        # Save the generation function
        self.generate_fn = generate_fn
        
        # Create main window
        self.root = tk.Tk()
        self.root.title("AI Chat Interface")
        self.root.geometry("600x800")
        self.root.configure(bg='white')
        
        # Create main container
        main_frame = tk.Frame(self.root, bg='white')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
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
    
    def generate_response(self):
        """Generate response based on input text."""
        prompt = self.input_text.get("1.0", tk.END).strip()
        if not prompt:
            return
            
        if not self.generate_fn:
            self.output_text.delete("1.0", tk.END)
            self.output_text.insert(tk.END, "No generation function provided")
            return
            
        try:
            # Disable button during generation
            self.generate_btn.config(state='disabled')
            self.generate_btn.update()
            
            # Generate response
            result = self.generate_fn(
                prompt=prompt,
                max_length=100,
                temperature=0.7
            )
            
            # Display response
            self.output_text.delete("1.0", tk.END)
            self.output_text.insert(tk.END, result.get("texts", [""])[0])
            
        except Exception as e:
            logger.error(f"Generation error: {str(e)}")
            self.output_text.delete("1.0", tk.END)
            self.output_text.insert(tk.END, f"Error: {str(e)}")
        finally:
            self.generate_btn.config(state='normal')
    
    def run(self):
        """Start the GUI."""
        self.root.mainloop()
