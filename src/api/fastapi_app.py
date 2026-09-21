import asyncio
import json
import time
from pathlib import Path
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from ..inference.engine import InferenceEngine
from ..models.registry import ModelRegistry
from ..data.fetcher import DATA_RAW_DIR

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
STATIC_DIR = ROOT_DIR / "static"
TEMPLATES_DIR = ROOT_DIR / "templates"

class PredictRequest(BaseModel):
    text: str = ""
    top_k: int = Field(default=5, ge=1, le=20)
    temperature: float = Field(default=0.75, ge=0.05, le=2.0)
    top_p: float = Field(default=0.9, ge=0.1, le=1.0)
    repetition_penalty: float = Field(default=1.2, ge=1.0, le=3.0)

class GenerateRequest(BaseModel):
    text: str = ""
    num_words: int = Field(default=5, ge=1, le=30)
    temperature: float = Field(default=0.75, ge=0.05, le=2.0)
    top_p: float = Field(default=0.9, ge=0.1, le=1.0)
    repetition_penalty: float = Field(default=1.25, ge=1.0, le=3.0)

class ModelSwitchRequest(BaseModel):
    model_id: str

def create_fastapi_app(default_model_id: Optional[str] = None) -> FastAPI:
    app = FastAPI(
        title="Neural Next-Word Prediction Studio API",
        version="2.1.0",
        description="High-performance async NLP engine with WebSocket streaming and Residual Recurrent Networks.",
    )

    registry = ModelRegistry()
    available = registry.list_models()
    if not default_model_id:
        if "harry_potter_lore_stacked_lstm" in available:
            default_model_id = "harry_potter_lore_stacked_lstm"
        elif "dostoevsky_notes_stacked_gru" in available:
            default_model_id = "dostoevsky_notes_stacked_gru"
        elif "dostoevsky_core_stacked_lstm" in available:
            default_model_id = "dostoevsky_core_stacked_lstm"
        elif "st_lstm" in available:
            default_model_id = "st_lstm"
        else:
            default_model_id = list(available.keys())[0] if available else "uni_lstm"

    engine = InferenceEngine(registry=registry, default_model_id=default_model_id)

    # Mount static directory
    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/", response_class=HTMLResponse)
    async def index():
        index_file = TEMPLATES_DIR / "index.html"
        if not index_file.exists():
            raise HTTPException(status_code=404, detail="Template not found")
        with open(index_file, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())

    @app.post("/api/predict")
    async def predict(req: PredictRequest):
        # Run CPU/GPU-bound tensor prediction in background threadpool to avoid event loop blocking
        result = await asyncio.to_thread(
            engine.predict_next_candidates,
            text=req.text,
            top_k=req.top_k,
            temperature=req.temperature,
            top_p=req.top_p,
            repetition_penalty=req.repetition_penalty,
        )
        return {"status": "success", "data": result}

    @app.post("/api/generate")
    async def generate(req: GenerateRequest):
        result = await asyncio.to_thread(
            engine.generate_sequence,
            seed_text=req.text,
            num_words=req.num_words,
            temperature=req.temperature,
            top_p=req.top_p,
            repetition_penalty=req.repetition_penalty,
        )
        return {"status": "success", "data": result}

    @app.get("/api/models")
    async def list_models():
        return {
            "status": "success",
            "active_model": engine.active_model_id,
            "models": engine.registry.list_models(),
        }

    @app.post("/api/model/switch")
    async def switch_model(req: ModelSwitchRequest):
        try:
            meta = await asyncio.to_thread(engine.switch_model, req.model_id)
            return {"status": "success", "data": meta}
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.get("/api/datasets")
    async def list_datasets():
        datasets = []
        if DATA_RAW_DIR.exists():
            for f in DATA_RAW_DIR.glob("*.txt"):
                datasets.append({
                    "name": f.stem,
                    "filename": f.name,
                    "size_kb": round(f.stat().st_size / 1024, 1),
                })
        return {"status": "success", "datasets": datasets}

    @app.get("/api/health")
    async def health():
        return {
            "status": "healthy",
            "server": "FastAPI ASGI",
            "active_model": engine.active_model_id,
            "cache_size": len(engine.cache),
            "timestamp": time.time(),
        }

    # Backward compatibility endpoint
    @app.post("/get_suggestions")
    async def legacy_get_suggestions(data: Dict[str, Any]):
        input_sentence = data.get("input_sentence", "")
        pred = await asyncio.to_thread(engine.predict_next_candidates, text=input_sentence, top_k=1)
        return {"suggestions": pred.get("top_word", "")}

    # WebSocket Real-Time Typing & Streaming Endpoints
    @app.websocket("/ws/stream")
    async def websocket_stream(websocket: WebSocket):
        await websocket.accept()
        try:
            while True:
                data_str = await websocket.receive_text()
                data = json.loads(data_str)
                seed_text = data.get("text", "")
                num_words = int(data.get("num_words", 5))
                temperature = float(data.get("temperature", 0.75))
                top_p = float(data.get("top_p", 0.9))
                repetition_penalty = float(data.get("repetition_penalty", 1.25))

                # Stream tokens
                for chunk in engine.generate_stream(
                    seed_text=seed_text,
                    num_words=num_words,
                    temperature=temperature,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                ):
                    await websocket.send_json({"type": "token", "data": chunk})
                    await asyncio.sleep(0.04)  # Natural typing stream rhythm

                await websocket.send_json({"type": "done", "status": "complete"})
        except WebSocketDisconnect:
            pass

    return app
