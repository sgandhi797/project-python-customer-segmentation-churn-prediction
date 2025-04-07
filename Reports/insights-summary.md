# Business Insights Summary

## Executive Summary

This report summarizes the key findings from an RFM-based customer segmentation and churn prediction model developed using e-commerce transaction data. Through this analysis, we identified behavioral segments of customers, developed an ML model to predict churn, and provided actionable recommendations to improve customer retention and profitability.

---

## Key Findings

### 1. Customer Segments (RFM Clustering)
- Customers were successfully grouped into 4 behavior-based clusters using K-Means.
- One cluster contained high-frequency, high-monetary value customers with low recency — these are top-tier loyal customers.
- Another cluster included customers with high spend but poor recency — these are candidates for re-engagement.

### 2. Churn Analysis
- Customers with a recency score greater than 180 days were labeled as likely churned.
- A Random Forest model trained on Recency, Frequency, and Monetary value achieved strong **recall**, identifying churned users effectively.
- Feature importance ranked **Recency** as the strongest indicator of churn.

### 3. Behavioral Patterns
- Loyal customers tended to purchase more frequently but with smaller transaction sizes.
- Inactive customers typically had one or two large purchases before disappearing.
- Seasonal spikes in customer activity were seen around holidays, particularly in November and December.

---

## Recommendations

1. **Target re-engagement campaigns** at customers who have high spending history but haven't purchased recently.
2. **Implement loyalty programs** for frequent and high-value customers to retain them.
3. **Use email or SMS marketing automation** to check in with at-risk customers before they hit the 180-day mark.
4. **Continue to train churn prediction models** regularly with updated data to refine targeting precision.

---

## Next Steps

- Integrate this model into CRM or marketing platforms.
- Enrich the dataset with customer demographics for deeper segmentation.
- Build interactive dashboards for real-time tracking of customer segments and churn risk.

---

Prepared by: **Sunny Gandhi**  
