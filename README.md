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
- The dataset was made by the **UC Irvine Machine Learning Repository** and contains transactions from an online retailer between 01/12/2010 and 09/12/2011.

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

- **RFM Analysis**
  - Calculated Recency, Frequency, and Monetary Value for each customer
  - Standardized features for clustering
- **Customer Segmentation (K-Means)**
  - Used the Elbow Method to find optimal number of clusters
  - Segmented customers into 4 behavior-based groups
- **Churn Prediction**
  - Labeled churned customers based on recency > 180 days
  - Trained a Random Forest classifier to predict churn
  - Evaluated model performance with precision, recall, F1-score

---

## 🖼 Visualizations

![clusters_plot](https://github.com/user-attachments/assets/9814109c-75a4-48a6-9253-b690bbefc9a8)

---

## 📌 Key Insights

---

## 🚀 How to Use

---

## 📚 Tools & Technologies

--- 

## 📄 License

- This project is licensed under the MIT License.
