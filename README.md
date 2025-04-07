# 👥 Customer Segmentation & Churn Prediction (Python Project)

> This project leverages Python to perform customer segmentation using RFM (Recency, Frequency, Monetary) analysis and predicts customer churn using machine learning techniques. The analysis is based on e-commerce transaction data.

---

## 📈 Project Overview

- Understanding customer behavior is vital for enhancing retention strategies. This project aims to:
  - **Segment customers** based on purchasing behavior using RFM analysis.
  - **Identify high-risk customers** likely to churn.
  - **Develop predictive models** to anticipate customer churn.

---

## 📦 Dataset

- **Source**: [Kaggle – E-Commerce Data](https://www.kaggle.com/datasets/carrie1/ecommerce-data)
- The dataset was originally created by the **UC Irvine Machine Learning Repository**.
- The dataset includes transactions from an online retailer between 01/12/2010 and 09/12/2011 and contains fields such as `InvoiceNo`, `StockCode`, `Description`, `Quantity`, `InvoiceDate`, `UnitPrice`, `CustomerID`, and `Country`.

---

## 💡 Key Steps

- 📊 **RFM Analysis**
  - Calculated Recency, Frequency, and Monetary Value for each customer
  - Standardized features for clustering
- 📈 **Customer Segmentation (K-Means)**
  - Used the Elbow Method to find optimal number of clusters
  - Segmented customers into 4 behavior-based groups
- 🔮 **Churn Prediction**
  - Labeled churned customers based on recency > 180 days
  - Trained a Random Forest classifier to predict churn
  - Evaluated model performance with precision, recall, F1-score

---

## 🔍 Key Python Queries

- `STEP 1: Load and Explore Dataset`
  ```python
  import pandas as pd
  import matplotlib.pyplot as plt
  import seaborn as sns
  import datetime
  ```

- `Read the Dataset`
  ```python
  df = pd.read_csv(r"C:\Users\sgand\OneDrive\Documents\Data Analysis\Python\Customer Segmentation and Churn 
  Prediction\ecommerce-data.csv")
  ```

- `Basic Cleanup`
  ```python
  df.dropna(subset=['CustomerID'], inplace=True)
  df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
  df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
  df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]
  ```

- `STEP 2: RFM Analysis`
  ```python
  snapshot_date = df['InvoiceDate'].max() + datetime.timedelta(days=1)
  rfm = df.groupby('CustomerID').agg({
      'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
      'InvoiceNo': 'nunique',
      'TotalPrice': 'sum'
  })
  rfm.rename(columns={
      'InvoiceDate': 'Recency',
      'InvoiceNo': 'Frequency',
      'TotalPrice': 'MonetaryValue'
  }, inplace=True)
  ```

- `STEP 3: Scaling and Clustering`
  ```python
  from sklearn.preprocessing import StandardScaler
  from sklearn.cluster import KMeans

  scaler = StandardScaler()
  rfm_scaled = scaler.fit_transform(rfm)
  ```

- `Elbow Method`
  ```python
  sse = {}
  for k in range(1, 10):
      kmeans = KMeans(n_clusters=k, random_state=1)
      kmeans.fit(rfm_scaled)
      sse[k] = kmeans.inertia_

  plt.figure(figsize=(8,5))
  plt.plot(list(sse.keys()), list(sse.values()), marker='o')
  plt.xlabel("Number of clusters")
  plt.ylabel("SSE")
  plt.title("Elbow Method for Optimal K")
  plt.show()
  ```

- `Apply KMeans with K=4`
  ```python
  kmeans = KMeans(n_clusters=4, random_state=1)
  rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)
  ```

- `Visualize Clusters`
  ```python
  sns.pairplot(rfm.reset_index(), hue='Cluster', palette='Set1', height=3)
  plt.savefig(r"C:\Users\sgand\OneDrive\Documents\Data Analysis\Python\Customer Segmentation and Churn 
  Prediction/clusters_plot.png")
  ```

- `STEP 4: Churn Prediction`
  ```python
  from sklearn.model_selection import train_test_split
  from sklearn.ensemble import RandomForestClassifier
  from sklearn.metrics import classification_report
  import joblib
  ```

- `Define Churn Label (Recency > 180 days)`
  ```python
  rfm['Churn'] = rfm['Recency'].apply(lambda x: 1 if x > 180 else 0)
  ```

- `Train/Test Split`
  ```python
  model = RandomForestClassifier()
  model.fit(X_train, y_train)
  ```

- `Evaluate`
  ```python
  y_pred = model.predict(X_test)
  print(classification_report(y_test, y_pred))
  ```

- `Save Model`
  ```python
  joblib.dump(model, r"C:\Users\sgand\OneDrive\Documents\Data Analysis\Python\Customer Segmentation and Churn  
  Prediction/customer-segmentation-churn_model.pkl")
  ```

---

## 📂 Project Structure

```text
project-python-customer-segmentation-churn-prediction/
├── Data/
│   └── ecommerce-data.csv                   # Original Kaggle dataset
│
├── Notebooks/
│   └── customer_segmentation.ipynb          # Complete Jupyter Notebook analysis
│
├── Source/
│   ├── data_preprocessing.py                # Cleaning, feature engineering, total price calculation
│   ├── rfm_analysis.py                      # Recency, Frequency, Monetary value calculation
│   ├── clustering.py                        # K-Means clustering logic and visualizations
│   └── churn_prediction.py                  # Model training, evaluation, and saving
│
├── Models/
│   └── churn_model.pkl                      # Trained Random Forest model
│
├── Visuals/
│   └── clusters_plot.png                    # Visualization of customer segments
│
├── Reports/
│   └── insights_summary.md                  # Business-style insights and summary report
│
├── requirements.txt                         # Python libraries needed to run this project
├── .gitignore                               # Ignore checkpoints, system files, and data
├── README.md                                # Full project overview and usage guide
└── LICENSE                                  # MIT License file
```

---

## 📊 Visualizations

![clusters_plot](https://github.com/user-attachments/assets/9814109c-75a4-48a6-9253-b690bbefc9a8)

---

## 📌 Key Insights

- High-value customers often have low recency (recent activity) and high frequency
- A cluster of customers showed high spend but long inactivity → ideal for retention targeting
- The churn prediction model achieved strong recall on identifying at-risk customers

---

## 🚀 How to Use

- Clone this repository:
  - `git clone https://github.com/sgandhi797/project-python-customer-segmentation-churn-prediction.git`
  - cd project-python-customer-segmentation-churn-prediction
- Install requirements:
  - Download Anaconda Navigator
  - Install Jupyter Notebook from the Navigator
- Open and run the notebook:
  - jupyter notebook/Project - Jupyter Notebook - Customer Segmentation and Churn Prediction.ipynb

---

## 📚 Tools & Technologies

- Python 3
- Pandas and NumPy for data handling
- Matplotlib and Seaborn for visualization
- Scikit-learn for clustering and classification
- Jupyter Notebook for interactive analysis

---

## 📄 License

- This project is licensed under the MIT License.
