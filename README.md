# Fraud Detection System

## Overview
This project focuses on building a **fraud detection system** that identifies fraudulent transactions using **machine learning models** and deploys them in a **cost-effective hybrid cloud infrastructure**. It aims to minimize false positives, optimize model deployment, and maintain scalability with minimal cost.

## Features
- **Supervised & Anomaly Detection Models** (XGBoost, LightGBM, Isolation Forest, Autoencoders)
- **Hybrid Cloud Storage** (Google BigQuery, AWS S3 Free Tier)
- **Real-Time Fraud Scoring API** (FastAPI, Docker, AWS Lightsail/GCP VM)
- **Automated Model Training & Monitoring** (MLflow, Apache Airflow, SHAP Explainability)
- **BI Dashboard for Fraud Trends** (Streamlit, Tableau Public)

## Tech Stack
### **Data Processing & Storage**
- **Google BigQuery / PostgreSQL** (Free-tier structured data storage)
- **Pandas / Dask** (Efficient local data preprocessing)
- **Kafka / Apache Flink** (Optional real-time streaming)

### **Model Development**
- **Python (XGBoost, LightGBM, Scikit-learn, TensorFlow/PyTorch)**
- **Optuna** (Hyperparameter tuning)
- **SHAP / LIME** (Model explainability)

### **Deployment & API**
- **FastAPI / Flask** (Lightweight model API hosting)
- **Docker / Kubernetes (Local for portfolio, cloud optional)**
- **AWS Lightsail / GCP Free Tier VM** (Cost-efficient cloud hosting)
- **Cloudflare Tunnel** (Free alternative to load balancers)

### **Monitoring & MLOps**
- **MLflow** (Model tracking & experiment management)
- **Apache Airflow** (Automated retraining pipeline - local setup)
- **Grafana + Prometheus** (Self-hosted API monitoring)

## Project Roadmap
### **Phase 1: Data Collection & Preprocessing (Weeks 1-2)**
- Extract transaction logs and fraud labels.
- Store structured data in **BigQuery**.
- Perform **ETL using Pandas/Dask**.

### **Phase 2: Model Development & Evaluation (Weeks 3-4)**
- Train **XGBoost, LightGBM, Isolation Forest, Autoencoders**.
- Use **SHAP for explainability**.
- Evaluate models with **F1-score, Recall, ROC-AUC**.

### **Phase 3: Model Optimization & Validation (Weeks 5-6)**
- Tune hyperparameters using **Optuna**.
- Validate model against past fraud cases.

### **Phase 4: API Deployment & Integration (Weeks 7-8)**
- Deploy model using **FastAPI**.
- Host API on **GCP Free Tier VM / AWS Lightsail**.
- Use **Docker & Cloudflare Tunnel** for cost-efficient API exposure.

### **Phase 5: Monitoring & Continuous Learning (Weeks 9+)**
- Automate retraining via **Apache Airflow**.
- Monitor fraud trends in **Streamlit/Tableau Public**.
- Store logs in **Google Drive/AWS S3 Free Tier**.

## How to Run Locally
### **1. Clone the Repository**
```bash
 git clone https://github.com/yourusername/fraud-detection-system.git
 cd fraud-detection-system
```

### **2. Set Up Virtual Environment**
```bash
 python -m venv venv
 source venv/bin/activate  # For Windows: venv\Scripts\activate
```

### **3. Install Dependencies**
```bash
 pip install -r requirements.txt
```

### **4. Train Model**
```bash
 python train.py
```

### **5. Run API Locally**
```bash
 uvicorn api:app --host 0.0.0.0 --port 8000
```

### **6. Test API Endpoint**
```bash
 curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d '{"amount": 500.0, "location": "US", "device_id": "xyz123"}'
```

## Deployment Instructions
### **Cloud Deployment (GCP/AWS Free Tier)**
```bash
# Build Docker Image
 docker build -t fraud-detection-api .

# Push to Docker Hub (optional)
 docker tag fraud-detection-api your-dockerhub-username/fraud-detection-api:latest
 docker push your-dockerhub-username/fraud-detection-api

# Deploy on GCP VM
 gcloud compute instances create-with-container fraud-api --container-image your-dockerhub-username/fraud-detection-api
```

## Future Improvements
- Implement **real-time fraud detection with Kafka / Apache Flink**.
- Expand **model retraining using online learning**.
- Improve **BI Dashboard for business insights**.

## License
This project is licensed under **MIT License**.

---
