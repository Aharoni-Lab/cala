import asyncio
import warnings
from contextlib import asynccontextmanager

from cala.gui.dependencies import get_config
from cala.main import run_pipeline

try:
    from fastapi import FastAPI
except ImportError:
    warnings.warn("yall need fastapi if you want the gui; install cala[gui]", stacklevel=2)
    FastAPI = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> None:
    config = await get_config()

    # Start the streamers in the background
    # modify runner.run_streamers to not block the event loop or run it in a background task
    asyncio.create_task(run_pipeline(config))

    yield


def get_app() -> FastAPI:
    from fastapi.staticfiles import StaticFiles

    from cala.gui.routes import get_frontend_dir, router

    app = FastAPI(lifespan=lifespan, debug=True)
    app.include_router(router)
    app.mount(path="/dist", app=StaticFiles(directory=get_frontend_dir() / "dist"), name="dist")

    return app
