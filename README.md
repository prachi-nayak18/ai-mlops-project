🚀 End-to-End MLOps Pipeline
A production-ready Machine Learning pipeline with experiment tracking, containerized deployment, and automated CI/CD — built to mirror real-world ML engineering workflows.
📌 Project Overview
This project demonstrates a complete MLOps lifecycle — from model training and experiment tracking to containerized deployment and CI/CD automation. Designed to replicate production-grade ML engineering practices used in the industry.
🏗️ Architecture
Raw Data → Data Preprocessing → Model Training → MLflow (Experiment Tracking) → Model Evaluation → Docker Container → Kubernetes Deployment → GitHub Actions (CI/CD) → Production API
✨ Features
✅ Experiment Tracking — Log metrics, parameters, and models using MLflow
✅ Containerization — Dockerized application for consistent environments
✅ Orchestration — Kubernetes deployment for scalability
✅ CI/CD Pipeline — Automated testing and deployment via GitHub Actions
✅ Production Ready — Modular, clean, and scalable codebase
🛠️ Tech Stack
Language: Python 3.10
ML Framework: Scikit-learn / TensorFlow
Experiment Tracking: MLflow
Containerization: Docker
Orchestration: Kubernetes
CI/CD: GitHub Actions
Version Control: Git & GitHub
⚙️ Setup & Installation
Prerequisites: Python 3.10+, Docker Desktop, Kubernetes (Minikube or kubectl)
Clone the Repository
git clone https://github.com/prachi-nayak18/ai-mlops-project.git
Install Dependencies
pip install -r requirements.txt
Train the Model
python src/train.py
View Experiments in MLflow
mlflow ui
Open http://localhost:5000
Build & Run Docker Container
docker build -t mlops-pipeline .
docker run -p 8000:8000 mlops-pipeline
Deploy to Kubernetes
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
📈 Results
Model Accuracy: 94%
Training Time: ~5 mins
Docker Image Size: ~1.2 GB
API Response Time: <200ms
🙋‍♀️ Author
Prachi Nayak
GitHub: https://github.com/prachi-nayak18
LinkedIn: https://www.linkedin.com/in/prachi-nayak-125002330
⭐ If you found this helpful, please star this repo!
