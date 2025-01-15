# Model Tuning Project
A research workspace for fine-tuning and experimenting with large language models (LLMs), focused on optimizing performance and exploring different training strategies.

## 🚀 Quick Start

1. Set up the environment:
```bash
chmod +x setup_venv.sh
./setup_venv.sh
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure your experiment:
```bash
cp config/example.yaml config/my_experiment.yaml
```

## 📁 Project Organization

```
model_tuning/
├── config/                     # Hydra configuration files
│   ├── base.yaml              # Default configuration
│   ├── model/                 # Model architectures
│   │   ├── gpt2.yaml
│   │   └── bert.yaml
│   └── tuning/               # Training strategies
│       ├── lora.yaml
│       └── peft.yaml
├── data/                      # Dataset management
├── experiments/              # Experiment tracking
│   └── runs/                 # Individual run outputs
├── models/                   # Model checkpoints
├── src/                     # Core implementation
└── utils/                   # Helper functions
```

## 🎯 Supported Features

- Fine-tuning strategies:
  - LoRA (Low-Rank Adaptation)
  - QLoRA (Quantized LoRA)
  - Full fine-tuning
- Model architectures:
  - GPT-2
  - BERT
  - T5

## 🛠️ Running Experiments

1. Start a basic fine-tuning run:
```bash
python src/train.py model=gpt2 tuning=lora
```

2. Custom training configuration:
```bash
python src/train.py --config-path config/my_experiment.yaml
```

3. Monitor training:
```bash
tensorboard --logdir experiments/runs
```

## 📊 Experiment Tracking

We use Weights & Biases for experiment tracking. Set up your credentials:

```bash
wandb login
```

View your experiments at: https://wandb.ai/your-username/model-tuning

## 🧪 Testing

Run the test suite:
```bash
pytest tests/ -v
```

## 📝 Contributing

1. Create a new branch for your feature
2. Implement changes and add tests
3. Submit a pull request

## 📚 References

- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [PEFT Documentation](https://huggingface.co/docs/peft)
- [Hydra Configuration](https://hydra.cc/docs/intro/)

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
