import os
os.environ['TK_SILENCE_DEPRECATION'] = '1'

import tkinter as tk
from tkinter import scrolledtext
import logging
from src.utils.logging_utils import setup_logging
import threading
from src.utils.generation_configs import GenerationConfig  # Import GenerationConfig
import tkinter.ttk as ttk  # Import ttk for Progressbar
import markdown  # Import markdown for converting Markdown to HTML
from src.utils.personality_configs import PersonalityConfig

logger = setup_logging()

class ChatInterface:
    def __init__(self, model=None, tokenizer=None, device='cpu', generate_fn=None, tts_engine=None):
        # Save the generation function
        self.generate_fn = generate_fn
        
        # Assign tokenizer to an instance variable
        self.tokenizer = tokenizer  # Added line
        
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
        
        # Add Generation Configuration Dropdown
        config_frame = tk.Frame(controls_frame, bg='white')
        config_frame.pack(side=tk.LEFT, padx=10, pady=5)
        tk.Label(config_frame, text="Configuration:", bg='white').pack(side=tk.LEFT)
        
        self.config_var = tk.StringVar(value="STANDARD_QUALITY")  # Default selection
        config_options = list(GenerationConfig.__annotations__.keys())
        tk.OptionMenu(config_frame, self.config_var, *config_options, command=self.update_selected_config).pack(side=tk.LEFT, padx=5)
        
        # Add Personalities Dropdown
        personality_frame = tk.Frame(controls_frame, bg='white')
        personality_frame.pack(side=tk.LEFT, padx=10, pady=5)
        tk.Label(personality_frame, text="Personality:", bg='white').pack(side=tk.LEFT)
        
        self.personality_var = tk.StringVar(value="Blank")  # Default selection
        personality_options = list(PersonalityConfig.PERSONALITY_OPTIONS.keys())
        tk.OptionMenu(personality_frame, self.personality_var, *personality_options, command=self.update_selected_personality).pack(side=tk.LEFT, padx=5)
        
        # Add Selected Configuration Frame
        selected_config_frame = tk.LabelFrame(main_frame, text="Selected Configuration", bg='white')
        selected_config_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.selected_config_labels = {}
        config_params = [
            "max_new_tokens",
            "temperature",
            "top_k",
            "top_p",
            "repetition_penalty",
            "no_repeat_ngram_size",
            "num_beams",
            "early_stopping",
            "min_length",
            "length_penalty",
            "DIRECT_RESPONSE"
        ]
        
        for i, param in enumerate(config_params):
            row = i // 5
            col = (i % 5) * 2
            
            tk.Label(
                selected_config_frame,
                text=f"{param.replace('_', ' ').title()}:",
                bg='white',
                font=('Arial', 10)
            ).grid(row=row, column=col, padx=5, pady=5, sticky='e')
            
            self.selected_config_labels[param] = tk.Label(
                selected_config_frame,
                text="",  # Will be updated dynamically
                bg='white',
                font=('Arial', 10, 'bold')
            )
            self.selected_config_labels[param].grid(row=row, column=col+1, padx=5, pady=5, sticky='w')
        
        # Initialize the selected configuration display
        self.update_selected_config(self.config_var.get())
        
        # Add Spinner (Progressbar) Widget
        spinner_frame = tk.Frame(main_frame, bg='white')
        spinner_frame.pack(pady=(0, 10))
        
        self.spinner = ttk.Progressbar(
            spinner_frame,
            mode='indeterminate',
            length=200
        )
        self.spinner.pack()
        self.spinner.stop()  # Ensure spinner is stopped initially
        
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
        
        # Replace HtmlFrame with ScrolledText for rendering Markdown
        self.output_text = scrolledtext.ScrolledText(
            main_frame,
            height=12,
            font=('Arial', 11),
            wrap=tk.WORD,
            state='disabled'  # Make it read-only
        )
        self.output_text.pack(fill=tk.BOTH, expand=True)
        
        # Define tags for Markdown formatting
        self.output_text.tag_configure("bold", font=("Arial", 11, "bold"))
        self.output_text.tag_configure("italic", font=("Arial", 11, "italic"))
        self.output_text.tag_configure("header1", font=("Arial", 16, "bold"))
        self.output_text.tag_configure("header2", font=("Arial", 14, "bold"))
        self.output_text.tag_configure("header3", font=("Arial", 12, "bold"))
        # Add more tags as needed
        
        # Add keyboard shortcut
        self.root.bind('<Control-Return>', lambda e: self.generate_response())
        
        self.personalities = PersonalityConfig.PERSONALITY_OPTIONS
        
        # Initialize selected personality
        self.selected_personality = PersonalityConfig.BLANK
    
    def update_selected_config(self, config_name):
        """Update the Selected Configuration display based on the selected configuration."""
        try:
            eos_token_id = self.tokenizer.eos_token_id if self.tokenizer else None
            generation_params = GenerationConfig.get_config(config_name, eos_token_id)
            
            for param, label in self.selected_config_labels.items():
                value = generation_params.get(param, "N/A")
                label.config(text=str(value))
        except Exception as e:
            logger.error(f"Failed to update selected configuration: {str(e)}")
            for label in self.selected_config_labels.values():
                label.config(text="Error")
    
    def update_selected_personality(self, personality_name):
        """Update the selected personality."""
        self.selected_personality = PersonalityConfig.PERSONALITY_OPTIONS.get(personality_name, "")
    
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
        # Start spinner
        self.spinner.start()
        self.generate_btn.config(state='disabled')
        thread = threading.Thread(target=self._generate_response_thread)
        thread.start()
    
    def parse_markdown(self, text):
        """Simple Markdown parser to apply text formatting."""
        self.output_text.config(state='normal')
        self.output_text.delete("1.0", tk.END)
        
        lines = text.split('\n')
        for line in lines:
            if line.startswith('# '):
                self.output_text.insert(tk.END, line[2:] + '\n', "header1")
            elif line.startswith('## '):
                self.output_text.insert(tk.END, line[3:] + '\n', "header2")
            elif line.startswith('### '):
                self.output_text.insert(tk.END, line[4:] + '\n', "header3")
            else:
                self.insert_formatted_text(line + '\n')
    
        self.output_text.config(state='disabled')
    
    def insert_formatted_text(self, text):
        """Insert text with bold and italic formatting."""
        words = text.split(' ')
        for word in words:
            if word.startswith('**') and word.endswith('**'):
                self.output_text.insert(tk.END, word[2:-2] + ' ', "bold")
            elif word.startswith('*') and word.endswith('*'):
                self.output_text.insert(tk.END, word[1:-1] + ' ', "italic")
            else:
                self.output_text.insert(tk.END, word + ' ')
    
    def _generate_response_thread(self):
        """Thread target for generating response."""
        try:
            user_input = self.input_text.get("1.0", tk.END).strip()
            
            if not user_input:
                return
                
            # Get selected personality
            personality_header = self.selected_personality if hasattr(self, 'selected_personality') else ""
            
            # Prepend personality to the user input if not blank
            if personality_header:
                prompt = f"{personality_header}\n{user_input}"
            else:
                prompt = user_input
            
            # Get selected configuration
            config_name = self.config_var.get()
            eos_token_id = self.tokenizer.eos_token_id if self.tokenizer else None
            generation_params = GenerationConfig.get_config(config_name, eos_token_id)
            
            result = self.generate_fn(
                prompt=prompt,
                **generation_params  # Apply selected configuration
            )
            
            # Update metrics with animation
            self.update_metrics(result["metrics"])
            
            # Convert Markdown to plain text
            markdown_content = result["texts"][0]
            
            # Parse and display Markdown content
            self.parse_markdown(markdown_content)
            
        except Exception as e:
            logger.error(f"Generation error: {str(e)}")
            error_message = f"Error: {str(e)}"
            self.parse_markdown(error_message)
        finally:
            # Stop spinner and re-enable generate button in the main thread
            self.root.after(0, self.stop_spinner)
    
    def stop_spinner(self):
        """Stop the spinner and re-enable the generate button."""
        self.spinner.stop()
        self.generate_btn.config(state='normal')
    
    def run(self):
        """Start the GUI."""
        self.root.mainloop()
    
    def select_personality(self, personality_name: str):
        """
        Select a personality and prepend it to the user prompt.
        """
        personality_header = self.personalities.get(personality_name, "")
        self.current_prompt = f"{personality_header}\n{self.user_input}"
