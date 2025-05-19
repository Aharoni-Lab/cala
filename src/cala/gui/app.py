import asyncio
import shutil
import warnings
from contextlib import asynccontextmanager

from cala.gui.dependencies import get_config, get_stream_dir
from cala.main import run_pipeline

try:
    from fastapi import FastAPI
except ImportError:
    warnings.warn("yall need fastapi if you want the gui; install cala[gui]", stacklevel=2)
    FastAPI = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> None:
    config = await get_config()
    stream_dir = await get_stream_dir()
    stream_dir.mkdir(exist_ok=True, parents=True)

    # Start the streamers in the background
    # modify runner.run_streamers to not block the event loop or run it in a background task
    asyncio.create_task(run_pipeline(config))

    yield

    shutil.rmtree(stream_dir)


def get_app() -> FastAPI:
    from cala.gui.routes import router

    app = FastAPI(lifespan=lifespan)
    app.include_router(router)
    return app
