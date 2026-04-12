from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from app.routes.stream import router as stream_router
from app.config import (
    APP_NAME,
    APP_VERSION,
    CORS_ORIGINS,
    OUTPUT_DIR,
    TASK_DATA_DIR,
    ensure_directories,
)
from app.routes.upload import router as upload_router
from app.routes.task import router as task_router
from app.routes.result import router as result_router
from app.routes.ws import router as ws_router

ensure_directories()

app = FastAPI(
    title=APP_NAME,
    version=APP_VERSION
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS if CORS_ORIGINS else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload_router)
app.include_router(task_router)
app.include_router(result_router)
app.include_router(ws_router)
app.include_router(stream_router)

app.mount("/static/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")
app.mount("/static/tasks", StaticFiles(directory=TASK_DATA_DIR), name="tasks")


@app.get("/")
async def root():
    return {
        "message": "Road Damage Detection Backend is running",
        "version": APP_VERSION
    }