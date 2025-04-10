## Project Structure

galaxy10_app/
├── data.py              # Downloads and processes the Galaxy10 dataset
├── model.py             # Defines the Lightning model
├── train.py             # CLI entry point: handles args, training, testing
├── utils.py             # Helper functions (e.g., checkpointing if needed)
├── environment.yml      # Conda env file with dependencies
└── README.md