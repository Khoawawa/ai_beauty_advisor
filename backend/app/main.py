import os
from fastapi import FastAPI
from contextlib import asynccontextmanager

from api.v1.endpoints import face_parsing
from core.advisor.face_parsing.onnx_inference import FaceParsingONNX
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Initializing Face Parsing Engine...")

    current_dir = Path(__file__).resolve().parent
    # project_root = current_dir.parent.parent
    model_path = os.path.join(current_dir, "core", "advisor","face_parsing", "weights", "resnet18.onnx")

    app.state.face_parsing_engine = FaceParsingONNX(model_path)
    print("Face Parsing Engine initialized successfully.")
    yield
    print("Shutting down Face Parsing Engine...")
    app.state.face_parsing_engine = None

app = FastAPI(title="Face Parsing API", version="1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(face_parsing.router, prefix="/api/v1", tags=["face_parsing"])


