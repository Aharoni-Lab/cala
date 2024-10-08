import argparse
import os
from cala.config import Config
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Cala custom configurations.")
    parser.add_argument(
        "--video-directory", type=Path, help="Path to the video directory"
    )
    parser.add_argument(
        "--video-files", nargs="*", type=Path, help="List of video file names"
    )
    parser.add_argument(
        "--data-directory", type=Path, help="Path to the data directory"
    )
    parser.add_argument("--data-name", help="Name of the data set")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--env-file", help="Path to custom .env file")

    args = parser.parse_args()

    # Collect command-line arguments that are not None and exclude 'env_file'
    cmd_args = {
        k: v for k, v in vars(args).items() if v is not None and k != "env_file"
    }

    # command-line argument or environment variable
    env_file = args.env_file or os.environ.get(".env")

    # Instantiate Settings with command-line arguments and custom env_file if provided
    if env_file:
        settings = Config(_env_file=env_file, **cmd_args)
    else:
        settings = Config(**cmd_args)

    print("Configurations:")
    print(f"Video Directory: {settings.video_directory}")
    print(f"Video Files: {settings.video_files}")
    print(f"Data Directory: {settings.data_directory}")
    print(f"Data Name: {settings.data_name}")
    print(f"Debug: {settings.debug}")


if __name__ == "__main__":
    main()
