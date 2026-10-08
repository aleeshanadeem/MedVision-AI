import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

# 1. Generate realistic medical/health dataset for risk prediction
np.random.seed(42)
n_samples = 1200

data = {
    'Age': np.random.randint(20, 75, n_samples),
    'BMI': np.random.uniform(18.5, 40.0, n_samples),
    'Blood_Pressure': np.random.randint(90, 180, n_samples),
    'Cholesterol': np.random.randint(150, 300, n_samples),
    'Glucose_Level': np.random.randint(70, 200, n_samples)
}

df = pd.DataFrame(data)

# Target Logic: High BP + High Cholesterol + High Glucose + High Age = Health Risk (1)
df['Risk_Status'] = (
    (df['Blood_Pressure'] > 140) & 
    (df['Cholesterol'] > 240) & 
    (df['Glucose_Level'] > 140)
).astype(int)

# Add some controlled noise for realism
noise = np.random.choice([0, 1], size=n_samples, p=[0.92, 0.08])
df['Risk_Status'] = np.bitwise_xor(df['Risk_Status'], noise)

X = df.drop('Risk_Status', axis=1)
y = df['Risk_Status']

# 2. ML Pipeline & Scaling
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Model Training
model = RandomForestClassifier(n_estimators=150, random_state=42)
model.fit(X_train_scaled, y_train)

# 4. Save Artifacts
joblib.dump(model, 'medvision_model.pkl')
joblib.dump(scaler, 'medvision_scaler.pkl')

print("MedVision AI Pipeline trained successfully!")
print("Model Accuracy:", model.score(X_test_scaled, y_test))