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
    print("  uv run monitor-model    - Run model monitoring pipeline")
    print("  uv run dashboard        - Run web dashboard")
    print()
    print("Environment Variables:")
    print("  BASE_DATA_PATH       - Path to data directory (required)")
    print("  WANDB_KEY           - Weights & Biases API key (for training)")
    print("  ENTRYPOINT_COMMAND  - Docker entrypoint command (deployment)")
    print()
    print("Local Usage:")
    print("  export BASE_DATA_PATH='/path/to/data'")
    print("  uv run batch-collect run")
    print("  uv run build-features run")
    print("  uv run train-model run")
    print()
    print("Docker/Cloud Deployment:")
    print("  Set ENTRYPOINT_COMMAND to: dashboard, batch-collect,")
    print("  build-features, train-model, or monitor-model")
    print("  Default: dashboard")
    print()
    print("For more information, see README.md")


if __name__ == "__main__":
    main()
