"""
Churn Prediction Script
- Defines churn based on Recency
- Trains and evaluates Random Forest model
"""

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

def label_churn(rfm, threshold=180):
    rfm['Churn'] = rfm['Recency'].apply(lambda x: 1 if x > threshold else 0)
    return rfm

def train_model(rfm):
    X = rfm[['Recency', 'Frequency', 'MonetaryValue']]
    y = rfm['Churn']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    
    joblib.dump(model, "models/churn_model.pkl")
    return model