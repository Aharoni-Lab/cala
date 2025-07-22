import asyncio

from cala.config import Config
from cala.core.execute import Executor
from cala.io import IO
from cala.util.new import package_frame


async def run_pipeline(config: Config) -> None:
    """Run the Cala processing pipeline.

    Args:
        config: User configurations
    """

    io = IO()
    stream = io.stream(config.input_files)
    runner = Executor(config)

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
