from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.config import CORS_ORIGINS, OUTPUTS_DIR, TASKS_DIR, UPLOADS_DIR
from app.routes.upload import router as upload_router
from app.routes.task import router as task_router
from app.routes.result import router as result_router
from app.routes.worker import router as worker_router
from app.routes.ws import router as ws_router

app = FastAPI(
    title="Road Damage Video Analysis API",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static/outputs", StaticFiles(directory=str(OUTPUTS_DIR)), name="outputs")
app.mount("/static/tasks", StaticFiles(directory=str(TASKS_DIR)), name="tasks")
app.mount("/static/uploads", StaticFiles(directory=str(UPLOADS_DIR)), name="uploads")

app.include_router(upload_router, prefix="/api")
app.include_router(task_router, prefix="/api")
app.include_router(result_router, prefix="/api")
app.include_router(worker_router, prefix="/api")
app.include_router(ws_router)

@app.get("/")
async def root():
    return {
        "message": "Road Damage Video Analysis API",
        "status": "ok"
    }