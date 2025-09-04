from contextlib import asynccontextmanager
from typing import AsyncGenerator, Any

from fastapi import FastAPI


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, Any]:
    yield


def get_app() -> FastAPI:
    from fastapi.staticfiles import StaticFiles
    from cala.gui.dependencies import get_frontend_dir
    from cala.gui.routes import router

    app = FastAPI(lifespan=lifespan, debug=True)

    app.mount(path="/dist", app=StaticFiles(directory=get_frontend_dir() / "dist"), name="dist")
    app.include_router(router)

    return app
