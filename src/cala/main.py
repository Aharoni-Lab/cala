import asyncio
from pathlib import Path

from cala.config import Config
from cala.connection_manager import ConnectionManager
from cala.io import IO
from cala.streaming.composer import Runner
from cala.streaming.util import package_frame


async def run_pipeline(config: Config, socket_manager: ConnectionManager, stream_dir: Path) -> None:
    """Run the Cala processing pipeline.

    Args:
        config: User configurations
        socket_manager: Connection manager for streaming connections.
    """

    io = IO()
    stream = io.stream(config.input_files)
    runner = Runner(config.pipeline, config.output_dir, socket_manager, stream_dir)

    try:
        for idx, frame in enumerate(stream):
            frame = package_frame(frame, idx)
            frame = runner.preprocess(frame)

            if not runner.is_initialized:
                runner.initialize(frame)
                continue

            await asyncio.to_thread(runner.iterate, frame)

    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
    except Exception as e:
        print(f"Error during processing: {str(e)}")
        raise
    finally:
        # close stream
        stream.close()
        # close resources and perform garbage collection
        runner.cleanup()
