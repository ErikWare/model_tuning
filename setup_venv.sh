#!/bin/bash

# =============================================================================
# Deployment Script: setup_env.sh
# Description: Sets up a local Python virtual environment, installs dependencies,
#              configures VS Code, and removes Conda dependencies.
# =============================================================================

# Exit immediately if a command exits with a non-zero status
set -e

# =============================================================================
# Configuration Variables
# =============================================================================

PROJECT_DIR="/Users/erikware/Desktop/model_tuning"
VENV_DIR="$PROJECT_DIR/venv"
REQUIREMENTS_FILE="$PROJECT_DIR/requirements.txt"
VS_CODE_SETTINGS="$HOME/Library/Application Support/Code/User/settings.json"
SHELL_CONFIG_FILES=("$HOME/.bashrc" "$HOME/.zshrc")  # Add other shell config files if needed

# =============================================================================
# Functions
# =============================================================================

# Function to create a virtual environment
create_virtual_env() {
    echo "Creating virtual environment at $VENV_DIR..."
    python3 -m venv "$VENV_DIR"
    echo "Virtual environment created."
}

# Function to activate the virtual environment
activate_virtual_env() {
    echo "Activating virtual environment..."
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
    echo "Virtual environment activated."
}

# Function to install dependencies
install_dependencies() {
    echo "Upgrading pip..."
    pip install --upgrade pip

    echo "Installing dependencies from requirements.txt..."
    pip install -r "$REQUIREMENTS_FILE"
    echo "Dependencies installed."
}

# Function to backup a file
backup_file() {
    local file_path="$1"
    if [ -f "$file_path" ]; then
        cp "$file_path" "${file_path}.bak_$(date +%s)"
        echo "Backup created for $file_path at ${file_path}.bak_$(date +%s)"
    fi
}

# Function to configure VS Code settings
configure_vscode() {
    echo "Configuring VS Code to use the new virtual environment..."

    # Backup existing settings.json
    backup_file "$VS_CODE_SETTINGS"

    # Define the Python interpreter path
    PYTHON_PATH="$VENV_DIR/bin/python"

    # Use jq to modify JSON if available, else use Python
    if command -v jq &> /dev/null; then
        echo "Using jq to modify VS Code settings..."
        jq --arg pythonPath "$PYTHON_PATH" \
           '(.python.pythonPath) = $pythonPath' \
           "$VS_CODE_SETTINGS" > "${VS_CODE_SETTINGS}.tmp" && mv "${VS_CODE_SETTINGS}.tmp" "$VS_CODE_SETTINGS"
    else
        echo "jq not found. Using Python to modify VS Code settings..."
        python3 - <<END
import json
import os

settings_path = os.path.expanduser("$VS_CODE_SETTINGS")
with open(settings_path, 'r') as f:
    settings = json.load(f)

settings["python.pythonPath"] = "$PYTHON_PATH"

with open(settings_path, 'w') as f:
    json.dump(settings, f, indent=4)
END
    fi

    echo "VS Code configured to use the virtual environment."
}

# Function to remove Conda dependencies from VS Code settings
remove_conda_from_vscode() {
    echo "Removing Conda dependencies from VS Code settings..."

    # Backup existing settings.json
    backup_file "$VS_CODE_SETTINGS"

    # Remove Conda-specific settings
    if command -v jq &> /dev/null; then
        echo "Using jq to remove Conda settings..."
        jq 'del(.python.condaPath, .python.condaEnvPath)' "$VS_CODE_SETTINGS" > "${VS_CODE_SETTINGS}.tmp" && mv "${VS_CODE_SETTINGS}.tmp" "$VS_CODE_SETTINGS"
    else
        echo "jq not found. Using Python to remove Conda settings..."
        python3 - <<END
import json
import os

settings_path = os.path.expanduser("$VS_CODE_SETTINGS")
with open(settings_path, 'r') as f:
    settings = json.load(f)

# Remove Conda-specific keys if they exist
settings.pop("python.condaPath", None)
settings.pop("python.condaEnvPath", None)

with open(settings_path, 'w') as f:
    json.dump(settings, f, indent=4)
END
    fi

    echo "Conda dependencies removed from VS Code settings."
}

# Function to remove Conda initialization from shell config files
remove_conda_from_shell() {
    echo "Removing Conda initialization from shell configuration files..."

    for config_file in "${SHELL_CONFIG_FILES[@]}"; do
        if [ -f "$config_file" ]; then
            echo "Processing $config_file..."
            # Backup the shell config file
            backup_file "$config_file"

            # Remove lines between Conda initialization markers
            sed -i.bak '/^# >>> conda initialize >>>/,/^# <<< conda initialize <<</d' "$config_file"

            echo "Conda initialization removed from $config_file."
        else
            echo "Shell config file $config_file does not exist. Skipping..."
        fi
    done

    echo "Conda initialization removed from all specified shell configuration files."
}

# Function to clean up Conda environments (optional)
cleanup_conda_envs() {
    echo "Listing all Conda environments..."
    conda env list

    echo "If you wish to remove any Conda environments, please do so manually using:"
    echo "conda remove --name <env_name> --all"
}

# Function to reload VS Code (requires VS Code to be closed and reopened)
reload_vscode() {
    echo "Please reload VS Code to apply the changes."
    echo "You can manually reload by pressing 'Cmd + Shift + P' (macOS) or 'Ctrl + Shift + P' (Windows/Linux) and selecting 'Reload Window'."
}

# =============================================================================
# Main Script Execution
# =============================================================================

echo "=== Starting Deployment Script ==="

# Step 1: Create Virtual Environment
create_virtual_env

# Step 2: Activate Virtual Environment
activate_virtual_env

# Step 3: Install Dependencies
install_dependencies

# Step 4: Configure VS Code to Use Virtual Environment
configure_vscode

# Step 5: Remove Conda Dependencies from VS Code Settings
remove_conda_from_vscode

# Step 6: Remove Conda Initialization from Shell Config Files
remove_conda_from_shell

# Step 7: (Optional) Cleanup Conda Environments
# Uncomment the following line if you wish to list Conda environments
# cleanup_conda_envs

# Step 8: Reload VS Code
reload_vscode

echo "=== Deployment Script Completed Successfully ==="

#!/bin/bash

# Exit on error
set -e

echo "🚀 Setting up virtual environment for ML project..."

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3 first."
    exit 1
fi

# Create and activate virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
echo "🔄 Upgrading pip..."
pip install --upgrade pip

# Install packages one by one with error handling
echo "📚 Installing dependencies..."

# Core packages first
echo "Installing scipy (this might take a while)..."
pip install scipy==1.13.1

echo "Installing numpy..."
pip install "numpy>=1.24.3,<1.25.0"

echo "Installing PyTorch..."
pip install torch==2.2.0

# Install remaining packages
packages=(
    "transformers==4.37.2"
    "accelerate==0.27.2"
    "openai-whisper==20231117"
    "sounddevice==0.4.6"
    "tqdm==4.66.2"
    "wandb==0.16.3"
)

for package in "${packages[@]}"; do
    echo "Installing $package..."
    pip install "$package" || {
        echo "❌ Failed to install $package"
        exit 1
    }
done

# Verify installation
echo "✅ Verifying installation..."
python3 -c "import torch; import transformers; import whisper; import sounddevice" || {
    echo "❌ Verification failed. Some packages are not properly installed."
    exit 1
}

echo "🎉 Setup completed successfully!"
echo "To activate the virtual environment, run: source venv/bin/activate"
