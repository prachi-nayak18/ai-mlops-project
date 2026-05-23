import logging
import os
import pickle
import time

import uvicorn
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="MLOps Model API",
    description="ML model serving endpoint",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_model = None
_preprocessor = None
_config = None
_start_time = time.time()


@app.on_event("startup")
async def load_artifacts():
    global _model, _preprocessor, _config
    config_path = os.getenv("CONFIG_PATH", "configs/config.yaml")
    try:
        with open(config_path) as f:
            _config = yaml.safe_load(f)
    except Exception as e:
        logger.warning(f"Config not found: {e}")

    model_path = _config["serving"]["model_path"] if _config else "models/artifacts/model.pkl"
    try:
        with open(model_path, "rb") as f:
            _model = pickle.load(f)
        logger.info(f"✅ Model loaded")
    except FileNotFoundError:
        logger.warning("⚠️  Model file not found — train first")


@app.get("/health")
def health():
    return {
        "status": "ok" if _model else "model not loaded",
        "model_loaded": _model is not None,
        "uptime_seconds": round(time.time() - _start_time, 1),
    }


@app.get("/model/info")
def model_info():
    if _model is None:
        return {"status": "No model loaded yet"}
    return {
        "model_type": type(_model).name,
        "task": _config["model"]["type"] if _config else "unknown",
        "algorithm": _config["model"]["algorithm"] if _config else "unknown",
    }


@app.post("/predict")
def predict(request: dict):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run pipeline first.")
    try:
        import pandas as pd
        df = pd.DataFrame([request["features"]])
        if _preprocessor:
            X = _preprocessor.transform(df)
        else:
            X = df.values
        prediction = _model.predict(X)[0]
        return {"prediction": int(prediction) if hasattr(prediction, 'item') else prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("src.serving.api:app", host="0.0.0.0", port=8000, reload=True)