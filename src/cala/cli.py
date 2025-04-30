import argparse


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Cala: A neural endoscope image processing tool")
    parser.add_argument(
        "--config",
        type=str,
        default="cala_config.yaml",
        help="Path to configuration file (default: cala_config.yaml)",
    )
    parser.add_argument(
        "--visual",
        action="store_true",
        help="Enable real-time processing visualization",
    )
    return parser.parse_args()
