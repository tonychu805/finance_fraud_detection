# Fraud Detection System Roadmap & Deployment Details

## 1. Roadmap

### Phase 1: Data Collection & Preprocessing (Weeks 1-2)
- Gather transaction logs, customer profiles, and fraud labels from internal sources.
- Use **Google BigQuery (free tier)** for structured storage and querying.
- Perform **ETL processing using Pandas/Dask** for cost-effective local data transformation.

### Phase 2: Model Development & Evaluation (Weeks 3-4)
- Perform **Exploratory Data Analysis (EDA)** using Pandas and Seaborn.
- Implement **supervised learning models** (XGBoost, LightGBM, Logistic Regression).
- Test anomaly detection methods (Isolation Forest, Autoencoders, One-Class SVM).
- Evaluate models using **Recall, F1-score, False Positive Rate, and ROC-AUC**.
- Track experiments with **MLflow (local setup or lightweight cloud storage like Google Drive/AWS S3 Free Tier).**

### Phase 3: Model Optimization & Validation (Weeks 5-6)
- Perform hyperparameter tuning using **Optuna**.
- Use **SHAP** for explainability to ensure transparency in fraud decisions.
- Conduct cross-validation and test against historical fraud cases.

### Phase 4: API Deployment & Integration (Weeks 7-8)
- Deploy model as a **REST API** using **FastAPI (lightweight & free to run on local or cloud VM).**
- Containerize with **Docker** for consistency.
- Use **AWS Lightsail or a lightweight GCP VM** for low-cost hosting.
- Expose API via **NGINX or Cloudflare Tunnel (free tiers available).**

### Phase 5: Monitoring & Continuous Learning (Weeks 9+)
- Set up a **BI Dashboard using Streamlit (self-hosted) or free Tableau Public.**
- Automate **weekly retraining with a cron job or Apache Airflow (local setup).**
- Store logs in **Google Drive/AWS S3 (free tier).**
- Monitor API usage with **Prometheus + Grafana (self-hosted on lightweight VM).**

---

## 2. Deployment Details

### Infrastructure
- **Hybrid Cloud Strategy:**
  - **Cloud:** Use **Google BigQuery** for querying, AWS/GCP free-tier VMs for hosting.
  - **Local:** Run development and training on **a local machine with GPU acceleration if available**.

- **Compute:** AWS Lightsail (cheaper alternative to EC2), GCP Free Tier VM
- **Storage:** Google BigQuery (limited free usage), AWS S3 (free tier for storage)
- **Model Serving:** FastAPI hosted on **GCP/AWS VM or local machine**
- **Orchestration:** Apache Airflow (local, no cloud cost)

### Deployment Steps
1. **Prepare Model for Deployment**
   - Train and save the best-performing model.
   - Convert the model into a deployable format (e.g., `.pkl`, `.onnx`).
   
2. **Create API for Model Serving**
   - Develop a RESTful API using **FastAPI**.
   - Define endpoints for fraud classification.
   - Implement request validation and logging mechanisms.
   
3. **Containerize the Application**
   - Write a **Dockerfile** to package the API.
   - Test locally before deployment.
   
4. **Deploy to Cloud Infrastructure (Hybrid Setup)**
   - Push Docker image to **Docker Hub (free tier)**.
   - Deploy on a **GCP Free Tier VM or AWS Lightsail** (low-cost hosting option).
   - Expose API via **Cloudflare Tunnel** (free alternative to costly load balancers).
   
5. **Monitor & Maintain**
   - Set up **Grafana + Prometheus (self-hosted) for API monitoring**.
   - Store logs in **Google Drive or AWS S3** (minimal cost).
   - Implement auto-retraining pipeline using **Apache Airflow (local setup).**

---

## 3. Key Considerations

### Security & Compliance
- **Data Encryption:** Use **SSL/TLS for API** and encrypt logs before cloud storage.
- **IAM Policies:** Restrict access using **GCP IAM & AWS IAM** (free policies).
- **Audit Logs:** Store transaction logs with **timestamped records**.

### Cost Optimization Strategies
- **Compute:** Use **GCP Free Tier VM or AWS Lightsail (low-cost alternative to EC2).**
- **Storage:** Store **only necessary logs in AWS S3/Google Drive (free tier limits).**
- **Model Retraining:** Automate using **local Airflow instead of managed ML services.**
- **API Hosting:** Use **Cloudflare Tunnel (free) instead of costly load balancers.**

### Business Impact & ROI Measurement
- Measure fraud detection **cost savings and chargeback reduction**.
- Track **false positive reduction rate** to improve customer experience.
- Analyze **performance trends over time using BI dashboards**.

---

## 4. Next Steps
- Begin **data collection and feature engineering**.
- Set up **development environment and CI/CD pipeline**.
- Develop **first iteration of fraud detection model**.
- Deploy **proof-of-concept API** for initial testing.
