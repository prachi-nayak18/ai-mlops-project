#!/bin/bash
set -e

echo "======================================"
echo "  MLOps Pipeline Runner"
echo "======================================"

CONFIG=${1:-configs/config.yaml}

echo "📥 Step 1: Installing dependencies..."
pip install -r requirements.txt -q

echo "🚀 Step 2: Running pipeline..."
python pipelines/pipeline.py --config $CONFIG

echo "📡 Step 3: Starting MLflow UI (background)..."
mlflow ui --host 0.0.0.0 --port 5000 &

echo "🌐 Step 4: Starting API server..."
uvicorn src.serving.api:app --host 0.0.0.0 --port 8000 --reload

echo "======================================"
echo "✅ All services running!"
echo "   MLflow : http://localhost:5000"
echo "   API    : http://localhost:8000/docs"
echo "======================================"