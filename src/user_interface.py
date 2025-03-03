import os
os.environ['TK_SILENCE_DEPRECATION'] = '1'

import tkinter as tk
from tkinter import scrolledtext
import logging
from src.utils.logging_utils import setup_logging
import threading
from src.utils.generation_configs import GenerationConfig  # Import GenerationConfig
import tkinter.ttk as ttk  # Import ttk for styling and Progressbar
import markdown  # Import markdown for converting Markdown to HTML
from src.utils.personality_configs import PersonalityConfig
from src.utils.markdown_formatter import MarkdownFormatter  # Import MarkdownFormatter
from src.utils.voice_to_text import VoiceToText  # Updated import
from src.utils.text_to_speech import TextToSpeech  # Updated import
import time
import queue
import json
from pathlib import Path

# =============================================================================
# Define a common color and font scheme for a modern, friendly look:
# =============================================================================
APP_BG = "#f0f2f5"         # Overall app background (light gray)
FRAME_BG = "white"         # Frame background (white for contrast)
ACCENT_COLOR = "#4A90E2"   # Accent blue for headings and highlights
BUTTON_BG = "#4A90E2"      # Primary button background
BUTTON_FG = "white"        # Primary button text color
ALT_BUTTON_BG = "#8BC34A"  # Alternate button background (e.g., for Load Model)
FONT_LABEL = ("Helvetica", 10)
FONT_LABEL_BOLD = ("Helvetica", 10, "bold")
FONT_BODY = ("Helvetica", 11)
FONT_HEADER = ("Helvetica", 12, "bold")

logger = setup_logging()

class ChatInterface:
    def __init__(self, model=None, tokenizer=None, device='cpu', generate_fn=None, tts_engine=None, models_list={}, controller_class=None, default_model_path=None):
        self.logger = logger  # Global logger
        
        # Save generation function and tokenizer
        self.generate_fn = generate_fn
        self.tokenizer = tokenizer

        # Create main window
        self.root = tk.Tk()
        self.root.title("AI Chat Interface")
        self.root.geometry("800x900")
        self.root.configure(bg=APP_BG)
        
        # Create main container frame
        main_frame = tk.Frame(self.root, bg=FRAME_BG)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # =========================================================================
        # Top Controls: Merged Model Selection & Generation Controls
        # =========================================================================
        top_controls_frame = tk.LabelFrame(main_frame, text="Model & Generation Controls", bg=FRAME_BG, fg=ACCENT_COLOR, font=FONT_HEADER)
        top_controls_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Left side: Generation Config & Personality
        gen_controls_frame = tk.Frame(top_controls_frame, bg=FRAME_BG)
        gen_controls_frame.pack(side=tk.LEFT, padx=10, pady=5)
        tk.Label(gen_controls_frame, text="Configuration:", bg=FRAME_BG, font=FONT_LABEL, fg="#333333").pack(side=tk.LEFT)
        self.config_var = tk.StringVar(value="STANDARD_QUALITY")
        config_options = list(GenerationConfig.__annotations__.keys())
        tk.OptionMenu(gen_controls_frame, self.config_var, *config_options, command=self.update_selected_config).pack(side=tk.LEFT, padx=5)
        
        tk.Label(gen_controls_frame, text="Personality:", bg=FRAME_BG, font=FONT_LABEL, fg="#333333").pack(side=tk.LEFT, padx=(10,0))
        self.personality_var = tk.StringVar(value="Blank")
        personality_options = list(PersonalityConfig.PERSONALITY_OPTIONS.keys())
        tk.OptionMenu(gen_controls_frame, self.personality_var, *personality_options, command=self.update_selected_personality).pack(side=tk.LEFT, padx=5)
        
        # Right side: Model Selection
        model_select_frame = tk.Frame(top_controls_frame, bg=FRAME_BG)
        model_select_frame.pack(side=tk.RIGHT, padx=10, pady=5)
        tk.Label(model_select_frame, text="Select Model:", bg=FRAME_BG, font=FONT_LABEL, fg="#333333").pack(side=tk.LEFT, padx=5)
        self.model_var = tk.StringVar()
        default_model = next(iter(models_list)) if models_list else "Default"
        self.model_var.set(default_model)
        tk.OptionMenu(model_select_frame, self.model_var, *list(models_list.keys())).pack(side=tk.LEFT, padx=5)
        # Use a ttk.Button for a rounded look:
        ttk.Button(model_select_frame, text="Load Model", command=self.reload_model, style="Rounded.TButton").pack(side=tk.LEFT, padx=5)
        
        # =========================================================================
        # Model Status Frame (will be updated on model load)
        # =========================================================================
        # Get initial model info
        model_name = getattr(model.config, '_name_or_path', 'Unknown Model')
        model_params = sum(p.numel() for p in model.parameters()) / 1_000_000
        
        status_frame = tk.LabelFrame(main_frame, text="Model Status", bg=FRAME_BG, fg=ACCENT_COLOR, font=FONT_HEADER)
        status_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.model_status_label = tk.Label(status_frame, text=f"Loaded Model: {model_name}", bg=FRAME_BG, font=FONT_LABEL_BOLD, fg="#333333")
        self.model_status_label.pack(anchor='w', pady=2)
        self.model_params_label = tk.Label(status_frame, text=f"Parameters: {model_params:.1f}M", bg=FRAME_BG, font=FONT_LABEL, fg="#333333")
        self.model_params_label.pack(anchor='w', pady=2)
        self.model_device_label = tk.Label(status_frame, text=f"Device: {device}", bg=FRAME_BG, font=FONT_LABEL, fg="#333333")
        self.model_device_label.pack(anchor='w', pady=2)
        
        # =========================================================================
        # Metrics Frame: Displays generation performance metrics with separate labels for names and values.
        # =========================================================================
        metrics_frame = tk.LabelFrame(main_frame, text="Metrics", bg=FRAME_BG, fg=ACCENT_COLOR, font=FONT_HEADER)
        metrics_frame.pack(fill=tk.X, pady=(0, 10))
        self.metrics_labels = {}
        for key in ["Generation Time", "Tokens/Second", "Input Tokens", "Output Tokens"]:
            metric_frame = tk.Frame(metrics_frame, bg=FRAME_BG)
            metric_frame.pack(side=tk.LEFT, padx=5, pady=5)
            # Label for metric name
            tk.Label(metric_frame, text=f"{key}:", bg=FRAME_BG, font=FONT_LABEL, fg="#333333").pack()
            # Label for metric value (to be updated later)
            self.metrics_labels[key] = tk.Label(metric_frame, text="N/A", bg=FRAME_BG, font=FONT_LABEL_BOLD, fg="#333333")
            self.metrics_labels[key].pack()
        
        # =========================================================================
        # Selected Configuration Display
        # =========================================================================
        selected_config_frame = tk.LabelFrame(main_frame, text="Selected Configuration", bg=FRAME_BG, fg=ACCENT_COLOR, font=FONT_HEADER)
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
            "length_penalty"
        ]
        for i, param in enumerate(config_params):
            row = i // 5
            col = (i % 5) * 2
            tk.Label(selected_config_frame, text=f"{param.replace('_', ' ').title()}:", bg=FRAME_BG, font=FONT_LABEL, fg="#333333").grid(row=row, column=col, padx=5, pady=5, sticky='e')
            self.selected_config_labels[param] = tk.Label(selected_config_frame, text="", bg=FRAME_BG, font=FONT_LABEL_BOLD, fg="#333333")
            self.selected_config_labels[param].grid(row=row, column=col+1, padx=5, pady=5, sticky='w')
        self.update_selected_config(self.config_var.get())
        
        # =========================================================================
        # Input Area
        # =========================================================================
        tk.Label(main_frame, text="Enter your prompt:", bg=FRAME_BG, font=FONT_LABEL, fg="#333333").pack(anchor='w')
        self.input_text = scrolledtext.ScrolledText(main_frame, height=6, font=FONT_BODY, wrap=tk.WORD)
        self.input_text.pack(fill=tk.X, pady=(0, 10))
        
        # =========================================================================
        # Actions Section: Combined buttons and spinner
        # =========================================================================
        actions_frame = tk.LabelFrame(main_frame, text="Actions", bg=FRAME_BG, fg=ACCENT_COLOR, font=FONT_HEADER)
        actions_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Set up a ttk style for round buttons
        style = ttk.Style()
        style.theme_use('default')
        style.configure("Rounded.TButton",
                        padding=6,
                        relief="flat",
                        background=BUTTON_BG,
                        foreground=BUTTON_FG,
                        font=FONT_BODY)
        style.map("Rounded.TButton",
                  background=[("active", ACCENT_COLOR)],
                  foreground=[("active", "white")])
        
        # Create a frame to hold action buttons
        actions_buttons_frame = tk.Frame(actions_frame, bg=FRAME_BG)
        actions_buttons_frame.pack(side=tk.LEFT, padx=10, pady=5)
        
        self.generate_btn = ttk.Button(actions_buttons_frame, text="Generate Response", command=self.generate_response, style="Rounded.TButton")
        self.generate_btn.grid(row=0, column=0, padx=5, pady=5)
        
        self.listen_button = ttk.Button(actions_buttons_frame, text="Hold to Speak", style="Rounded.TButton")
        self.listen_button.grid(row=0, column=1, padx=5, pady=5)
        self.listen_button.bind('<ButtonPress-1>', self.start_recording)
        self.listen_button.bind('<ButtonRelease-1>', self.stop_recording)
        
        # In the Actions Section, create a persistent variable for TTS checkbox
        self.tts_var = tk.BooleanVar(value=True)  # <-- New persistent TTS state variable
        self.tts_checkbox = ttk.Checkbutton(
            actions_buttons_frame,
            text="Enable TTS",
            variable=self.tts_var,
            command=self.toggle_tts
        )
        self.tts_checkbox.grid(row=0, column=2, padx=5, pady=5)
        
        self.stop_tts_button = ttk.Button(actions_buttons_frame, text="Stop TTS", command=lambda: self.text_to_speech.stop(), style="Rounded.TButton")
        self.stop_tts_button.grid(row=0, column=3, padx=5, pady=5)
        
        self.tts_test_button = ttk.Button(actions_buttons_frame, text="Test TTS", command=self.test_text_to_speech, style="Rounded.TButton")
        self.tts_test_button.grid(row=0, column=4, padx=5, pady=5)
        
        # Add spinner (status indicator) in the Actions section (to the right)
        self.spinner = ttk.Progressbar(actions_frame, mode='indeterminate', length=100, style="TProgressbar")
        self.spinner.pack(side=tk.RIGHT, padx=10, pady=5)
        self.spinner.stop()
        
        # =========================================================================
        # Output Area
        # =========================================================================
        tk.Label(main_frame, text="Response:", bg=FRAME_BG, font=FONT_LABEL, fg="#333333").pack(anchor='w')
        self.output_text = scrolledtext.ScrolledText(main_frame, height=12, font=FONT_BODY, wrap=tk.WORD, state='disabled')
        self.output_text.pack(fill=tk.BOTH, expand=True)
        
        # Initialize MarkdownFormatter and tag configurations
        self.markdown_formatter = MarkdownFormatter(self.output_text)
        self.output_text.tag_configure("bold", font=("Helvetica", 11, "bold"))
        self.output_text.tag_configure("italic", font=("Helvetica", 11, "italic"))
        self.output_text.tag_configure("header1", font=("Helvetica", 16, "bold"))
        self.output_text.tag_configure("header2", font=("Helvetica", 14, "bold"))
        self.output_text.tag_configure("header3", font=("Helvetica", 12, "bold"))
        
        # =========================================================================
        # Keyboard Shortcut for Generation
        # =========================================================================
        self.root.bind('<Control-Return>', lambda e: self.generate_response())
        
        # Other initializations
        self.personalities = PersonalityConfig.PERSONALITY_OPTIONS
        self.selected_personality = PersonalityConfig.BLANK
        self.voice_to_text = VoiceToText()
        self.text_to_speech = TextToSpeech()
        self.text_to_speech.enabled = True
        
        # GUI update queue
        self.gui_queue = queue.Queue()
        self.root.after(100, self.process_gui_queue)
        
        self.device = device
        self.controller_class = controller_class
        self.current_model_path = default_model_path
        self.models_list = models_list  # Expected as a dict { "ModelName": "path", ... }
    
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
        """Update the Selected Configuration display based on the chosen configuration."""
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
        """
        Update the metrics display with the generation performance.

        Expected metrics keys:
          - 'generation_time': Time taken to generate the response.
          - 'tokens_per_second': Generation speed.
          - 'input_tokens': Number of tokens in the prompt.
          - 'new_tokens': List or collection of output tokens.
        """
        self.metrics_labels["Generation Time"].config(text=f"{metrics['generation_time']:.2f}s")
        self.metrics_labels["Tokens/Second"].config(text=f"{metrics['tokens_per_second']:.1f} t/s")
        self.metrics_labels["Input Tokens"].config(text=str(metrics['input_tokens']))
        self.metrics_labels["Output Tokens"].config(text=str(sum(metrics['new_tokens'])))
        # Briefly highlight metrics for improved user feedback
        for label in self.metrics_labels.values():
            label.config(foreground=ACCENT_COLOR)
            self.root.after(1000, lambda l=label: l.config(foreground="#333333"))
    
    def generate_response(self):
        """Generate response asynchronously using a background thread.
        
        This method retrieves the user's input, constructs the prompt by including 
        any selected personality, retrieves generation parameters, performs asynchronous 
        generation, and enqueues UI update tasks including metrics update and TTS processing.
        """
        self.logger.debug("generate_response: Starting asynchronous generation.")
        self.spinner.start()  # Start spinner to indicate processing
        self.generate_btn.config(state='disabled')  # Disable button to prevent re-entry
        
        def worker():
            try:
                # Retrieve and validate user input
                user_input = self.input_text.get("1.0", tk.END).strip()
                if not user_input:
                    self.logger.debug("generate_response.worker: No input provided.")
                    self.gui_queue.put((self.parse_markdown, ("Error: No input provided.",)))
                    return

                # Build the prompt by prepending the selected personality if available
                personality_header = self.selected_personality if hasattr(self, 'selected_personality') else ""
                prompt = f"{personality_header}\n{user_input}" if personality_header else user_input

                # Get generation parameters from configuration
                config_name = self.config_var.get()
                eos_token_id = self.tokenizer.eos_token_id if self.tokenizer else None
                generation_params = GenerationConfig.get_config(config_name, eos_token_id)

                self.logger.debug("generate_response.worker: Invoking generate_fn asynchronously.")
                # Call the generation function to produce the model response
                result = self.generate_fn(prompt=prompt, **generation_params)
                self.logger.debug("generate_response.worker: Generation successful.")

                # Enqueue metrics update and response handling (with markdown parsing and TTS)
                self.gui_queue.put((self.update_metrics, (result["metrics"],)))
                markdown_content = result["texts"][0]
                self.gui_queue.put((self.handle_generated_response, (markdown_content,)))
            except Exception as e:
                self.logger.error(f"generate_response.worker: Generation error: {e}", exc_info=True)
                self.gui_queue.put((self.parse_markdown, (f"Error: {e}",)))
            finally:
                self.gui_queue.put((self.stop_spinner, ()))
        
        # Start the asynchronous worker thread
        threading.Thread(target=worker, daemon=True).start()

    def handle_generated_response(self, markdown_content):
        """
        Handle markdown conversion and TTS invocation after response generation.
        """
        self.parse_markdown(markdown_content)
        if self.text_to_speech.enabled:
            # Run TTS asynchronously on a background thread
            threading.Thread(target=self.run_tts, daemon=True).start()
    
    def parse_markdown(self, text):
        """Parse and display Markdown content."""
        self.markdown_formatter.parse_markdown(text)
    
    def insert_formatted_text(self, text):
        """Insert text with formatting."""
        words = text.split(' ')
        for word in words:
            if word.startswith('**') and word.endswith('**'):
                self.output_text.insert(tk.END, word[2:-2] + ' ', "bold")
            elif word.startswith('*') and word.endswith('*'):
                self.output_text.insert(tk.END, word[1:-1] + ' ', "italic")
            else:
                self.output_text.insert(tk.END, word + ' ')
    
    def run_tts(self):
        """
        Run TTS on the current output content.
        """
        # Retrieve the content from the output field
        current_text = self.output_text.get("1.0", tk.END).strip()
        if current_text:
            self.text_to_speech.speak(current_text)
    
    def stop_spinner(self):
        """Stop the spinner and re-enable the generate button."""
        self.spinner.stop()
        self.generate_btn.config(state='normal')
    
    def run(self):
        """Start the GUI event loop."""
        self.root.mainloop()
    
    def start_recording(self, event):
        """Start voice recording."""
        self.logger.info("Hold to Speak button pressed. Starting recording.")
        self.input_text.delete("1.0", tk.END)
        self.voice_to_text.start_recording()
    
    def stop_recording(self, event):
        """Stop recording and process transcription."""
        self.logger.info("Hold to Speak button released. Stopping recording.")
        def handle_transcription():
            text = self.voice_to_text.stop_recording()
            if text:
                self.logger.info(f"Transcription received: {text}")
                self.input_text.insert(tk.END, text + '\n')
        threading.Thread(target=handle_transcription, daemon=True).start()
    
    def on_close(self):
        """
        Clean up resources on exit.
        """
        self.logger.info("Application shutting down. Stopping background tasks...")
        self.voice_to_text.stop_listening()
        self.stop_tts_playback()
        self.root.destroy()
    
    def toggle_tts(self):
        """Toggle Text-to-Speech based on checkbox selection."""
        # Use the persistent self.tts_var instead of an undefined one
        self.text_to_speech.enabled = self.tts_var.get()
        if not self.text_to_speech.enabled:
            self.text_to_speech.stop()
    
    def stop_tts_playback(self):
        """Stop any ongoing TTS playback."""
        try:
            self.text_to_speech.stop()
        except Exception as e:
            self.logger.error(f"Error stopping TTS playback: {e}")
    
    def test_text_to_speech(self):
        """Test the text-to-speech functionality using input text."""
        text = self.input_text.get("1.0", tk.END).strip()
        if text:
            self.text_to_speech.speak(text)
        else:
            self.logger.warning("No text available for Text-to-Speech.")
    
    def reload_model(self):
        """
        Reload and recreate the model entirely by loading the configuration from config.yaml.
        """
        selected_name = self.model_var.get()
        # Import load_model_config from main.py to use as the single source for model paths
        from src.main import load_model_config
        models_config = load_model_config()
        if selected_name not in models_config:
            self.logger.error("Selected model not found in config.yaml.")
            return
        model_rel_path = models_config[selected_name]
        # Use same logic as in main.py: remove leading "./" if present and prepend models folder
        if model_rel_path.startswith("./"):
            model_rel_path = model_rel_path[2:]
        project_root = Path(__file__).parents[1]
        new_model_path = project_root / "models" / model_rel_path

        self.logger.info(f"Reloading model: {selected_name} from {new_model_path}")
        if hasattr(self, "controller"):
            self.logger.info("Destroying existing model controller")
            del self.controller

        controller = self.controller_class(
            model_path=new_model_path,
            device=self.device
        )
        self.controller = controller
        self.generate_fn = controller.generate
        self.tokenizer = controller.tokenizer
        
        new_model_name = getattr(controller.model.config, '_name_or_path', 'Unknown Model')
        new_model_params = sum(p.numel() for p in controller.model.parameters()) / 1_000_000
        self.model_status_label.config(text=f"Loaded Model: {new_model_name}")
        self.model_params_label.config(text=f"Parameters: {new_model_params:.1f}M")
        self.model_device_label.config(text=f"Device: {self.device}")
        self.logger.info(f"Model reloaded: {selected_name}")