"""
Create a simple test model for deployment verification.
"""
import pickle
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Create a simple dummy model
X = np.random.rand(100, 4)  # 100 samples, 4 features
y = np.random.randint(0, 2, 100)  # Binary classification

# Train a simple model
model = RandomForestClassifier(n_estimators=10, random_state=42)
model.fit(X, y)

# Save the model
with open('models/fraud_detector.pkl', 'wb') as f:
    pickle.dump(model, f)

print("✅ Test model created successfully!") 