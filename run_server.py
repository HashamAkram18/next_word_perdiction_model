import sys
import argparse
import uvicorn

# Ensure UTF-8 output on Windows console
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.api.fastapi_app import create_fastapi_app

def main():
    parser = argparse.ArgumentParser(description="Neural Next-Word Prediction Studio (FastAPI ASGI).")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host address")
    parser.add_argument("--port", type=int, default=5050, help="Port to listen on")
    parser.add_argument("--model", type=str, default=None, help="Initial model to load")
    parser.add_argument("--reload", action="store_true", help="Auto-reload on code change")
    args = parser.parse_args()

    app = create_fastapi_app(default_model_id=args.model)
    print("\n=======================================================")
    print(" [Neural Predict Studio] FastAPI ASGI Engine Active")
    print(f" Access URL: http://{args.host}:{args.port}")
    print(" Swagger Docs: http://{args.host}:{args.port}/docs")
    print(" WebSocket Stream: ws://{args.host}:{args.port}/ws/stream")
    print(" Ultra-Fast Non-Blocking Inference & Residual Networks")
    print("=======================================================\n")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")

if __name__ == "__main__":
    main()
