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
- **Created By**: UC Irvine Machine Learning Repository
- The dataset contains transactions from an online retailer between 01/12/2010 and 09/12/2011 and contains fields such as `InvoiceNo`, `StockCode`, `Description`, `Quantity`, `InvoiceDate`, `UnitPrice`, `CustomerID`, and `Country`.

---

## 📂 Project Structure

```text
customer-segmentation-churn-prediction/
├── data/
│   └── ecommerce-data.csv
├── notebooks/
│   └── customer_segmentation.ipynb
├── src/
│   └── *.py (scripts)
├── visuals/
├── models/
├── requirements.txt
└── README.md
```

---

## 🔍 Key Steps

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

## 🖼 Visualizations

![clusters_plot](https://github.com/user-attachments/assets/9814109c-75a4-48a6-9253-b690bbefc9a8)

---

## 📌 Key Insights

- High-value customers often have low recency (recent activity) and high frequency
- A cluster of customers showed high spend but long inactivity → ideal for retention targeting
- The churn prediction model achieved strong recall on identifying at-risk customers

---

## 🚀 How to Use

- Clone this repository:
  - git clone https://github.com/sgandhi797/project-python-customer-segmentation-churn-prediction.git
  - cd project-python-customer-segmentation-churn-prediction
- Install requirements:
  - Download Anaconda Navigator
  - Install Jupyter Notebook from the Navigator
- Open and run the notebook:
  - xxx  

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
