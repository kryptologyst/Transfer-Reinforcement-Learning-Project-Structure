#!/usr/bin/env python3
"""Setup script for transfer reinforcement learning project."""

import subprocess
import sys
from pathlib import Path


def run_command(command, description):
    """Run a command and handle errors."""
    print(f"Running: {description}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed:")
        print(f"  Error: {e.stderr}")
        return False


def main():
    """Main setup function."""
    print("Transfer Reinforcement Learning Project Setup")
    print("=" * 50)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("✗ Python 3.8+ is required")
        sys.exit(1)
    print(f"✓ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    
    # Install dependencies
    if not run_command("pip install -r requirements.txt", "Installing dependencies"):
        print("Failed to install dependencies. Please check requirements.txt")
        sys.exit(1)
    
    # Create necessary directories
    directories = [
        "assets/plots",
        "assets/videos",
        "assets/models",
        "assets/results",
        "assets/reports",
        "data",
        "checkpoints",
        "logs",
        "tensorboard_logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {directory}")
    
    # Run tests
    if not run_command("python -m pytest tests/ -v", "Running tests"):
        print("Some tests failed, but setup can continue")
    
    print("\n" + "=" * 50)
    print("Setup completed successfully!")
    print("\nNext steps:")
    print("1. Run the complete example: python scripts/complete_example.py")
    print("2. Train source agent: python scripts/train_source.py --config configs/cartpole.yaml")
    print("3. Run transfer learning: python scripts/transfer_learning.py --config configs/mountain_car.yaml")
    print("4. Launch demo: streamlit run demo/app.py")
    print("\nFor more information, see README.md")


if __name__ == "__main__":
    main()
