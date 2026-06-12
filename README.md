 # 🚀 End-to-End MLOps Pipeline > A production-ready Machine Learning pipeline with experiment tracking, containerized deployment, and automated CI/CD — built to mirror real-world ML engineering workflows.
 ![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python) ![MLflow](https://img.shields.io/badge/MLflow-Tracking-orange?logo=mlflow) ![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?logo=docker) ![Kubernetes](https://img.shields.io/badge/Kubernetes-Orchestration-326CE5?logo=kubernetes) ![GitHub Actions](https://img.shields.io/badge/CI/CD-GitHub_Actions-2088FF?logo=githubactions) ---
 ## 📌 Project Overview 
 This project demonstrates a complete MLOps lifecycle — from model training and experiment tracking to containerized deployment and CI/CD automation. Designed to replicate production-grade ML engineering practices used in the industry. ---
 ## 🏗️ Architecture
 Raw Data ↓ Data Preprocessing ↓ Model Training ──────→ MLflow (Experiment Tracking) ↓ Model Evaluation ↓ Docker Container ↓ Kubernetes Deployment ↓ GitHub Actions (CI/CD) ↓ Production API ---
 ## ✨ Features
 - ✅ Experiment Tracking — Log metrics, parameters, and models using MLflow
 - ✅ Containerization — Dockerized application for consistent environments
 - ✅ Orchestration — Kubernetes deployment for scalability
 - ✅ CI/CD Pipeline — Automated testing and deployment via GitHub Actions
 - ✅ Production Ready — Modular, clean, and scalable codebase ---
 -  ## 🛠️ Tech Stack
 | Category | Tools | |----------|-------| | Language | Python 3.10 | | ML Framework | Scikit-learn / TensorFlow | | Experiment Tracking | MLflow | | Containerization | Docker | | Orchestration | Kubernetes | | CI/CD | GitHub Actions | | Version Control | Git & GitHub |
   ## 📁 Project Structure
 mlops-pipeline/ │ ├── data/ │ ├── raw/ # Raw dataset │ └── processed/ # Preprocessed data │ ├── src/ │ ├── data_preprocessing.py # Data cleaning & feature engineering │ ├── train.py # Model training script │ ├── evaluate.py # Model evaluation │ └── predict.py # Inference script │ ├── mlflow/ │ └── mlruns/ # MLflow experiment logs │ ├── docker/ │ ├── Dockerfile # Docker image configuration │ └── docker-compose.yml # Multi-container setup │ ├── k8s/ │ ├── deployment.yaml # Kubernetes deployment config │ └── service.yaml # Kubernetes service config │ ├── .github/ │ └── workflows/ │ └── ci-cd.yml # GitHub Actions workflow │ ├── requirements.txt └── README.md 
 
 ## ⚙️ Setup & Installation
 ### Prerequisites - Python 3.10+ - Docker Desktop - Kubernetes (Minikube or kubectl) 
 ### 1. Clone the Repository
bash git clone https://github.com/prachi-nayak18/ai-mlops-project.git cd ai-mlops-project ### 2. Install Dependencies
bash pip install -r requirements.txt ### 3. Train the Model
bash python src/train.py ### 4. View Experiments in MLflow
bash mlflow ui # Open http://localhost:5000 ### 5. Build & Run Docker Container
bash docker build -t mlops-pipeline . docker run -p 8000:8000 mlops-pipeline ### 6. Deploy to Kubernetes
bash kubectl apply -f k8s/deployment.yaml kubectl apply -f k8s/service.yaml --- 

## 📊 MLflow Experiment Tracking MLflow tracks every training run with: - Parameters — learning rate, epochs, batch size - Metrics — accuracy, loss, F1-score - Artifacts — saved model files
python import mlflow with mlflow.start_run(): mlflow.log_param("learning_rate", 0.01) mlflow.log_metric("accuracy", 0.94) mlflow.sklearn.log_model(model, "model") ---

## 🐳 Docker Configuration
dockerfile FROM python:3.10-slim WORKDIR /app COPY requirements.txt . RUN pip install -r requirements.txt COPY . . CMD ["python", "src/predict.py"] --- 
## 📈 Results 
| Metric | Value | |--------|-------| | Model Accuracy | 94% | | Training Time | ~5 mins | | Docker Image Size | ~1.2 GB | | API Response Time | <200ms | --- 

## 🙋‍♀️ Author Prachi Nayak - 🔗 GitHub: [@prachi-nayak18](https://github.com/prachi-nayak18) - 💼 LinkedIn:[prachi-nayak-125002330](https://www.linkedin.com/in/prachi-nayak-125002330) ---

⭐ If you found this helpful, please star this repo! EOF
