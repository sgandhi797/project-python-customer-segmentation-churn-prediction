"""
Data Preprocessing Script
- Handles missing values
- Converts data types
- Calculates total transaction value
"""

import pandas as pd

def preprocess_data(df):
    df = df.dropna(subset=['CustomerID'])
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
    df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]
    return df