# Model Tuning Project
An experimental area for tuning different open source models for test and learn

## Virtual Environment Setup

1. Make the setup script executable:
```bash
chmod +x setup_venv.sh
```

2. Create and activate the virtual environment:
```bash
./setup_venv.sh
```

3. To deactivate the virtual environment when finished:
```bash
deactivate
```

Note: The virtual environment files will be stored in the `venv` directory. Make sure to add `venv/` to your `.gitignore` if you're using git.

## Enhanced Project Structure
```
model_tuning/
├── config/                     # Hydra configuration files
│   ├── config.yaml            # Base configuration
│   ├── model/                 # Model-specific configs
│   └── training/              # Training configs
├── data/                      # Training and evaluation data
│   ├── raw/                   # Original data
│   └── processed/             # Processed datasets
├── models/                    # Saved model checkpoints
│   └── gpt2/                 # GPT-2 specific models
├── notebooks/                 # Jupyter notebooks for exploration
├── scripts/                   # Training and utility scripts
├── src/                      # Source code
│   ├── data/                 # Data processing utilities
│   ├── models/               # Model implementations
│   ├── training/             # Training loops and utilities
│   └── utils/                # Helper functions
└── tests/                    # Unit tests
```

## Experiment Tracking
This project uses Weights & Biases for experiment tracking. Set up your credentials:
```bash
wandb login
```

## Running Experiments
Start a new training run:
```bash
python src/train.py model=gpt2 training=default
```

## Downloading Models
To download GPT-2 (117M) model and save it locally:
```bash
python scripts/download_gpt2.py
```

The model will be saved in `models/gpt2/` with the following structure:
```
models/gpt2/
├── model/              # Model weights and configuration
├── tokenizer/          # Tokenizer files
└── config.json         # Model configuration
```

## Testing
Run tests:
```bash
pytest tests/
```
