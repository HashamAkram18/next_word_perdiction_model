import time
from pathlib import Path
from flask import Flask, render_template, request, jsonify
from typing import Optional

from ..inference.engine import InferenceEngine
from ..models.registry import ModelRegistry
from ..data.fetcher import DATA_RAW_DIR

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
TEMPLATES_DIR = ROOT_DIR / "templates"
STATIC_DIR = ROOT_DIR / "static"

def create_app(default_model_id: Optional[str] = None) -> Flask:
    app = Flask(
        __name__,
        template_folder=str(TEMPLATES_DIR),
        static_folder=str(STATIC_DIR),
    )

    registry = ModelRegistry()
    # Choose default model: prefer newly trained harry_potter_lore or dostoevsky if present, else first available
    available = registry.list_models()
    if not default_model_id:
        if "harry_potter_lore_stacked_lstm" in available:
            default_model_id = "harry_potter_lore_stacked_lstm"
        elif "dostoevsky_core_stacked_lstm" in available:
            default_model_id = "dostoevsky_core_stacked_lstm"
        elif "st_lstm" in available:
            default_model_id = "st_lstm"
        elif "uni_lstm" in available:
            default_model_id = "uni_lstm"
        else:
            default_model_id = list(available.keys())[0] if available else "uni_lstm"

    engine = InferenceEngine(registry=registry, default_model_id=default_model_id)

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/api/predict", methods=["POST"])
    def predict():
        data = request.get_json(force=True, silent=True) or {}
        text = data.get("text", "") or data.get("input_sentence", "")
        top_k = int(data.get("top_k", 5))
        temperature = float(data.get("temperature", 0.7))

        try:
            result = engine.predict_next_candidates(text=text, top_k=top_k, temperature=temperature)
            return jsonify({"status": "success", "data": result})
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500

    @app.route("/api/generate", methods=["POST"])
    def generate():
        data = request.get_json(force=True, silent=True) or {}
        seed_text = data.get("text", "") or data.get("seed_text", "")
        num_words = int(data.get("num_words", 5))
        temperature = float(data.get("temperature", 0.7))

        try:
            result = engine.generate_sequence(seed_text=seed_text, num_words=num_words, temperature=temperature)
            return jsonify({"status": "success", "data": result})
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500

    @app.route("/api/models", methods=["GET"])
    def list_models():
        models_dict = engine.registry.list_models()
        return jsonify({
            "status": "success",
            "active_model": engine.active_model_id,
            "models": models_dict,
        })

    @app.route("/api/model/switch", methods=["POST"])
    def switch_model():
        data = request.get_json(force=True, silent=True) or {}
        model_id = data.get("model_id")
        if not model_id:
            return jsonify({"status": "error", "message": "Missing 'model_id' parameter"}), 400

        try:
            meta = engine.switch_model(model_id)
            return jsonify({"status": "success", "data": meta})
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 400

    @app.route("/api/datasets", methods=["GET"])
    def list_datasets():
        datasets = []
        if DATA_RAW_DIR.exists():
            for f in DATA_RAW_DIR.glob("*.txt"):
                datasets.append({
                    "name": f.stem,
                    "filename": f.name,
                    "size_kb": round(f.stat().st_size / 1024, 1),
                })
        return jsonify({"status": "success", "datasets": datasets})

    @app.route("/api/health", methods=["GET"])
    def health():
        return jsonify({
            "status": "healthy",
            "active_model": engine.active_model_id,
            "cache_size": len(engine.cache),
            "timestamp": time.time(),
        })

    # Legacy endpoint compatibility
    @app.route("/get_suggestions", methods=["POST"])
    def legacy_get_suggestions():
        data = request.get_json(force=True, silent=True) or {}
        input_sentence = data.get("input_sentence", "")
        pred = engine.predict_next_candidates(text=input_sentence, top_k=1)
        return jsonify({"suggestions": pred.get("top_word", "")})

    return app
