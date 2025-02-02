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
from src.utils.markdown_formatter import MarkdownFormatter  # Import MarkdownFormatter
from src.utils.speech_utils import VoiceToText  # Import VoiceToText
from src.utils.speech_utils import TextToSpeech  # Import TextToSpeech
import time  # Added import
import queue  # Added import

logger = setup_logging()

class ChatInterface:
    def __init__(self, model=None, tokenizer=None, device='cpu', generate_fn=None, tts_engine=None):
        self.logger = logger  # Assign the global logger to the instance
        
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
        
        # Initialize MarkdownFormatter
        self.markdown_formatter = MarkdownFormatter(self.output_text)
        
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

        # Initialize VoiceToText
        self.voice_to_text = VoiceToText()

        # Add "Hold to Speak" button
        listen_frame = tk.Frame(main_frame, bg='white')
        listen_frame.pack(pady=(0, 10))
        
        self.listen_button = tk.Button(
            listen_frame,
            text="Hold to Speak",
            font=('Arial', 11),
            pady=5,
            bg='lightblue'
        )
        self.listen_button.pack()
        
        # Bind press and release events
        self.listen_button.bind('<ButtonPress-1>', self.start_recording)  # Updated method
        self.listen_button.bind('<ButtonRelease-1>', self.stop_recording)  # Updated method

        # Initialize TextToSpeech - keep it simple
        self.text_to_speech = TextToSpeech()
        self.text_to_speech.enabled = True
        
        # Add a toggle button for TTS
        tts_toggle_frame = tk.Frame(main_frame, bg='white')
        tts_toggle_frame.pack(pady=(0, 10))
        
        self.tts_var = tk.BooleanVar(value=True)
        self.tts_checkbox = tk.Checkbutton(
            tts_toggle_frame,
            text="Enable Text-to-Speech",
            variable=self.tts_var,
            command=self.toggle_tts,
            bg='white'
        )
        self.tts_checkbox.pack()

        # Add "Stop TTS" button
        stop_tts_button = tk.Button(
            tts_toggle_frame,
            text="Stop TTS",
            command=lambda: self.text_to_speech.stop(),
            bg='white'
        )
        stop_tts_button.pack()
        
        # Add a button to test the text-to-speech functionality
        tts_test_frame = tk.Frame(main_frame, bg='white')
        tts_test_frame.pack(pady=(0, 10))
        
        self.tts_test_button = tk.Button(
            tts_test_frame,
            text="Test Text-to-Speech",
            command=self.test_text_to_speech,
            font=('Arial', 11),
            pady=5
        )
        self.tts_test_button.pack()
        
        # Initialize a queue for thread-safe GUI updates
        self.gui_queue = queue.Queue()
        
        # Start the GUI update loop
        self.root.after(100, self.process_gui_queue)
    
    def process_gui_queue(self):
        """Process queued GUI updates."""
        try:
            while True:
                func, args = self.gui_queue.get_nowait()
                func(*args)
        except queue.Empty:
            pass
        self.root.after(100, self.process_gui_queue)
    
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
        """Parse and display Markdown content using MarkdownFormatter."""
        self.markdown_formatter.parse_markdown(text)
    
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
        """
        This thread handles text generation asynchronously, preventing UI blocking.
        We also manage TTS in a separate daemon thread to avoid resource leaks.
        """
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
            
            # Update metrics via the GUI queue
            self.gui_queue.put((self.update_metrics, (result["metrics"],)))
            
            # Get Markdown content
            markdown_content = result["texts"][0]
            
            # Speak the generated text if TTS is enabled
            if self.text_to_speech.enabled:
                self.text_to_speech.speak(markdown_content)
            
            # Schedule Markdown parsing in the main thread
            self.gui_queue.put((self.parse_markdown, (markdown_content,)))
            
        except Exception as e:
            logger.error(f"Generation error: {str(e)}")
            error_message = f"Error: {str(e)}"
            self.gui_queue.put((self.parse_markdown, (error_message,)))
        finally:
            # Schedule spinner stop in the main thread
            self.gui_queue.put((self.stop_spinner, ()))
    
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

    def start_recording(self, event):
        """Handle button press to start recording."""
        self.logger.info("Hold to Speak button pressed. Starting to record.")
        # Clear the input_text
        self.input_text.delete("1.0", tk.END)

        # Start recording
        self.voice_to_text.start_recording()

    def stop_recording(self, event):
        """Handle button release to stop recording and process transcription."""
        self.logger.info("Hold to Speak button released. Stopping recording.")
        
        def handle_transcription():
            text = self.voice_to_text.stop_recording()
            if text:
                self.logger.info(f"Transcription received: {text}")
                self.input_text.insert(tk.END, text + '\n')
        
        # Handle transcription in a separate thread to avoid blocking the UI
        threading.Thread(target=handle_transcription, daemon=True).start()

    def on_close(self):
        """
        Cleans up resources (stopping recordings, destroying the window).
        Ensures any background threads terminate gracefully.
        """
        self.logger.info("Application is shutting down. Stopping background listening...")
        # ...existing cleanup code...
        self.voice_to_text.stop_listening()  # Ensure all listening is stopped
        self.stop_tts_playback()  # Stop any ongoing TTS
        self.root.destroy()
    
    def toggle_tts(self):
        """Simple toggle for TTS state."""
        self.text_to_speech.enabled = self.tts_var.get()
        if not self.text_to_speech.enabled:
            self.text_to_speech.stop()

    def stop_tts_playback(self):
        """
        Interrupt ongoing TTS playback if the user wants to stop early.
        """
        try:
            self.text_to_speech.stop()
        except Exception as e:
            self.logger.error(f"Error stopping TTS playback: {e}")
    
    def test_text_to_speech(self):
        """Test the text-to-speech functionality with the content in the user input section."""
        text = self.input_text.get("1.0", tk.END).strip()
        if text:
            self.text_to_speech.speak(text)
        else:
            self.logger.warning("No text available for Text-to-Speech.")
