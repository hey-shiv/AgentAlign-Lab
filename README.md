# AgentAlign-Lab

A laboratory for aligning agents through preference learning and direct preference optimization (DPO).

## Overview

AgentAlign-Lab provides a complete pipeline for:
- Task generation and management
- Agent trajectory collection and scoring
- Preference pair creation from trajectories
- Model training with DPO and SFT
- Comprehensive evaluation and analysis
- Interactive dashboard for result visualization

## Quick Start

```bash
# Install dependencies
uv sync

# Generate tasks
python scripts/01_generate_tasks.py

# Run agent
python scripts/02_run_agent.py

# Score trajectories
python scripts/03_score_trajectories.py

# Build preferences
python scripts/04_build_preferences.py

# Train model
python scripts/05_train_dpo.py

# Evaluate
python scripts/06_eval_models.py

# Launch dashboard
python scripts/07_launch_dashboard.py
```

## Project Structure

```
agentalign-lab/
├── configs/           # Configuration files (YAML)
├── data/             # Data storage (tasks, trajectories, preferences)
├── outputs/          # Model outputs (adapters, evals, figures)
├── report/           # Results and paper
├── src/agentalign/   # Main package
├── scripts/          # Pipeline scripts
├── tests/            # Test suite
└── notebooks/        # Jupyter notebooks
```

## Requirements

- Python 3.11+
- Modern package management (uv or pdm)

## Development

Run tests:
```bash
pytest tests/ -v
```

Format code:
```bash
ruff format src/ tests/
ruff check src/ tests/ --fix
```

## License

MIT
