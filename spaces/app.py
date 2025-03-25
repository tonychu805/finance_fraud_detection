import gradio as gr
import pandas as pd
import pickle
from huggingface_hub import hf_hub_download

# Download the model from Hugging Face Hub
model_path = hf_hub_download(
    repo_id="finance-fraud-detection",
    filename="model/fraud_detector.pkl"
)

# Load the model
with open(model_path, 'rb') as f:
    model = pickle.load(f)

def predict_fraud(amount, transaction_type, merchant_category, zip_code):
    """Predict if a transaction is fraudulent."""
    # Create a DataFrame with the input features
    data = pd.DataFrame({
        'amount': [float(amount)],
        'transaction_type': [transaction_type],
        'merchant_category': [merchant_category],
        'zip_code': [zip_code]
    })
    
    # Make prediction
    prediction = model.predict(data)[0]
    probability = model.predict_proba(data)[0][1]
    
    # Format result
    result = "🚨 Fraudulent" if prediction == 1 else "✅ Legitimate"
    confidence = f"{probability:.2%}"
    
    return result, confidence

# Create Gradio interface
iface = gr.Interface(
    fn=predict_fraud,
    inputs=[
        gr.Number(label="Transaction Amount ($)"),
        gr.Dropdown(
            choices=["online", "in-store", "atm"],
            label="Transaction Type"
        ),
        gr.Dropdown(
            choices=["retail", "entertainment", "grocery", "travel"],
            label="Merchant Category"
        ),
        gr.Text(label="ZIP Code")
    ],
    outputs=[
        gr.Text(label="Prediction"),
        gr.Text(label="Confidence")
    ],
    title="Finance Fraud Detection",
    description="Detect potentially fraudulent financial transactions using machine learning.",
    examples=[
        [1000.00, "online", "retail", "10001"],
        [50.00, "in-store", "grocery", "90210"],
        [5000.00, "atm", "travel", "60601"]
    ],
    theme="huggingface"
)

# Launch the app
iface.launch() 