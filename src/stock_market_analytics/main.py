#!/usr/bin/env python3
"""
Main entry point for the stock market analytics application.

This module provides a command-line interface to the various components
of the stock market analytics pipeline.
"""


def main() -> None:
    """
    Main entry point for the stock market analytics CLI.

    Displays usage information and available commands.
    """
    print("📈 Stock Market Analytics")
    print("=" * 50)
    print()
    print("Available Commands:")
    print("  uv run batch-collect    - Run data collection pipeline")
    print("  uv run build-features   - Run feature engineering pipeline")
    print("  uv run train-model      - Run model training pipeline")
    print()
    print("Environment Variables:")
    print("  BASE_DATA_PATH  - Path to data directory (required)")
    print("  WANDB_KEY       - Weights & Biases API key (for training)")
    print()
    print("Example Usage:")
    print("  export BASE_DATA_PATH='/path/to/data'")
    print("  uv run batch-collect run")
    print("  uv run build-features run")
    print("  uv run train-model run")
    print()
    print("For more information, see README.md")


if __name__ == "__main__":
    main()
