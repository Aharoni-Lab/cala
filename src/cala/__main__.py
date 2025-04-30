from pathlib import Path

from cala.cli import parse_args
from cala.config import Config
from cala.io import IO
from cala.streaming.composer import Runner
from cala.streaming.util import package_frame


def run_pipeline(config_path: Path, enable_visual: bool = False) -> None:
    """Run the Cala processing pipeline.

    Args:
        config_path: Path to the YAML configuration file
        enable_visual: Whether to enable real-time visualization
    """
    config = Config.from_yaml(config_path)

    io = IO()
    stream = io.stream(config.input_files)
    runner = Runner(config.pipeline, config.output_dir)

    if enable_visual:
        # visualization setup
        pass

    try:
        for idx, frame in enumerate(stream):
            frame = package_frame(frame, idx)
            frame = runner.preprocess(frame)

            if not runner.is_initialized:
                runner.initialize(frame)
                continue

            runner.iterate(frame)

    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
    except Exception as e:
        print(f"Error during processing: {str(e)}")
        raise
    finally:
        if enable_visual:
            # visualization cleanup
            pass

        # close stream
        stream.close()
        # close resources and perform garbage collection
        runner.cleanup()


def main() -> None:
    """Main entry point for Cala."""
    args = parse_args()
    config_path = Path(args.config)

    run_pipeline(config_path, args.visual)


if __name__ == "__main__":
    main()
