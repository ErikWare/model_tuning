
import tkinter as tk

class MarkdownFormatter:
    def __init__(self, text_widget):
        self.text_widget = text_widget
        self.configure_tags()

    def configure_tags(self):
        """Configure text tags for formatting."""
        self.text_widget.tag_configure("bold", font=("Arial", 11, "bold"))
        self.text_widget.tag_configure("italic", font=("Arial", 11, "italic"))
        self.text_widget.tag_configure("header1", font=("Arial", 16, "bold"))
        self.text_widget.tag_configure("header2", font=("Arial", 14, "bold"))
        self.text_widget.tag_configure("header3", font=("Arial", 12, "bold"))
        self.text_widget.tag_configure("code", font=("Courier", 10), background="#f0f0f0")
        # Add more tags as needed for syntax highlighting

    def parse_markdown(self, text):
        """Parse Markdown text and apply formatting."""
        self.text_widget.config(state='normal')
        self.text_widget.delete("1.0", tk.END)
        
        lines = text.split('\n')
        for line in lines:
            if line.startswith('# '):
                self.text_widget.insert(tk.END, line[2:] + '\n', "header1")
            elif line.startswith('## '):
                self.text_widget.insert(tk.END, line[3:] + '\n', "header2")
            elif line.startswith('### '):
                self.text_widget.insert(tk.END, line[4:] + '\n', "header3")
            elif line.startswith('```'):
                language = line[3:].strip()
                self.handle_code_blocks(lines, language)
                break
            else:
                self.insert_formatted_text(line + '\n')
        
        self.text_widget.config(state='disabled')

    def insert_formatted_text(self, text):
        """Insert text with bold and italic formatting."""
        words = text.split(' ')
        for word in words:
            if word.startswith('**') and word.endswith('**'):
                self.text_widget.insert(tk.END, word[2:-2] + ' ', "bold")
            elif word.startswith('*') and word.endswith('*'):
                self.text_widget.insert(tk.END, word[1:-1] + ' ', "italic")
            else:
                self.text_widget.insert(tk.END, word + ' ')

    def handle_code_blocks(self, lines, language):
        """Handle and format code blocks with syntax highlighting."""
        code_block = []
        for line in lines:
            if line.startswith('```'):
                break
            code_block.append(line)
        
        code_text = '\n'.join(code_block)
        self.text_widget.insert(tk.END, code_text + '\n', "code")