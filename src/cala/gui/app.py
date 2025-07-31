import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI


@asynccontextmanager
async def lifespan(app: FastAPI) -> None:
    asyncio.create_task()

    yield


def get_app() -> FastAPI:
    from fastapi.staticfiles import StaticFiles

    from cala.gui.routes import get_frontend_dir, router

    app = FastAPI(lifespan=lifespan, debug=True)

    app.mount(path="/dist", app=StaticFiles(directory=get_frontend_dir() / "dist"), name="dist")
    app.include_router(router)

    return app
