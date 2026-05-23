# 🚀 End-to-End MLOps Pipeline

A production-grade MLOps pipeline supporting any ML model with 
automated data ingestion, training, tracking, serving & deployment.



![Python](https://img.shields.io/badge/Python-3.11-blue)




![MLflow](https://img.shields.io/badge/MLflow-2.11-orange)




![FastAPI](https://img.shields.io/badge/FastAPI-0.110-green)




![Docker](https://img.shields.io/badge/Docker-Ready-blue)



---

## 🏗️ Architecture
Raw Data → Ingestion → Validation → Preprocessing
→ Training → MLflow Tracking → FastAPI Serving
→ Docker → Kubernetes → GitHub Actions CI/CD

## ⚙️ Tech Stack

| Component | Tool |
|-----------|------|
| ML Models | Scikit-learn, XGBoost |
| Tracking | MLflow |
| API | FastAPI + Uvicorn |
| Container | Docker + Kubernetes |
| CI/CD | GitHub Actions |

## 🚀 Quick Start

`bash
# Install
pip install -r requirements.txt

# Run Pipeline
python pipelines/pipeline.py --config configs/config.yaml

# MLflow UI
mlflow ui --port 5000

# API Server
uvicorn src.serving.api:app --port 8000 --reload

📊 Results
val_accuracy: 0.58
val_f1: 0.5597
Pipeline runs in under 30 seconds

📁 Project Structure
ai-mlops-project/
├── src/
│   ├── ingestion/    # Data loading & validation
│   ├── training/     # Model training + MLflow
│   ├── serving/      # FastAPI endpoints
│   └── monitoring/   # Drift detection
├── pipelines/        # End-to-end orchestrator
├── configs/          # Central config
├── docker/           # Dockerfiles
├── k8s/              # Kubernetes manifests
└── .github/          # CI/CD workflows

🤝 Connect
Open to AI/ML Engineer & MLOps roles 
Linkedin - https://www.linkedin.com/in/prachi-nayak-125002330?utm_source=share_via&utm_content=profile&utm_medium=member_android
