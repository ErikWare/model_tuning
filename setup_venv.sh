#!/bin/bash

# Exit on error
set -e

# Get absolute path of script directory
PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
VENV_DIR="$PROJECT_DIR/venv"
REQUIREMENTS_FILE="$PROJECT_DIR/requirements.txt"

echo "🔍 Project directory detected as: $PROJECT_DIR"

# Functions
create_virtual_env() {
    echo "📦 Creating virtual environment at $VENV_DIR..."
    python3 -m venv "$VENV_DIR"
    echo "Virtual environment created."
}

activate_virtual_env() {
    echo "🔄 Activating virtual environment..."
    source "$VENV_DIR/bin/activate"
    echo "Virtual environment activated."
}

install_dependencies() {
    echo "🔄 Upgrading pip..."
    pip install --upgrade pip

    echo "📚 Installing dependencies..."
    # Core packages first
    echo "Installing core dependencies..."
    pip install scipy numpy torch || {
        echo "❌ Failed to install core dependencies"
        exit 1
    }

    echo "Installing remaining packages..."
    pip install -r "$REQUIREMENTS_FILE" || {
        echo "❌ Failed to install remaining dependencies"
        exit 1
    }
}

verify_installation() {
    echo "✅ Verifying installation..."
    python3 -c "import torch; import transformers; import whisper; import sounddevice" || {
        echo "❌ Verification failed. Some packages are not properly installed."
        exit 1
    }
}

# Main Script Execution
echo "🚀 Setting up virtual environment for ML project..."

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3 first."
    exit 1
fi

create_virtual_env
activate_virtual_env
install_dependencies
verify_installation

echo "🎉 Setup completed successfully!"
echo "To activate the virtual environment, run: source venv/bin/activate"
