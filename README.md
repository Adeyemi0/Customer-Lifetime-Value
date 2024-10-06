# Customer Lifetime Value Prediction

## Table of Contents
- [Background Overview](#background-overview)
- [Data Dictionary](#data-dictionary)
- [Executive Summary](#executive-summary)
- [Insights Deep Dive](#insights-deep-dive)
- [Recommendations](#recommendations)
- [Further Studies and Limitations](#further-studies-and-limitations)

---

## Background Overview
Understanding the value that customers bring to a business over their lifetime is crucial for making informed marketing and sales decisions. In this project, we estimate the **Customer Lifetime Value (CLTV)** to help businesses identify high-value customers and allocate resources effectively.

The data used includes customer transactions and their details such as purchase quantities, product prices, and countries. The goal is to predict CLTV based on these factors and guide business decisions on customer segmentation and retention strategies.

---

## Data Dictionary
| Column         | Description                                      |
|----------------|--------------------------------------------------|
| **Invoice**    | Invoice number (unique identifier for transactions) |
| **StockCode**  | Product code (unique identifier for products)      |
| **Description**| Product description                               |
| **Quantity**   | Quantity of the product purchased                 |
| **InvoiceDate**| Date and time when the invoice was generated      |
| **Price**      | Price of the product per unit                     |
| **CustomerID** | Unique identifier for each customer               |
| **Country**    | Country where the customer resides                |

---

## Executive Summary
The project aims to estimate **Customer Lifetime Value (CLTV)** for each customer to help businesses understand how much value each customer will bring over time. We used various machine learning models and evaluated their performance to select the best one for this prediction task. The Random Forest model gave the best performance with a **Mean Squared Error (MSE)** of **36.7979**.

### Why CLTV Matters for Business
CLTV is a key metric because it helps companies:
- **Identify high-value customers:** Focusing on retaining these customers can drive long-term profitability.
- **Allocate marketing budgets more effectively:** By targeting customers likely to bring more value over time, businesses can optimize their advertising spending.
- **Improve retention strategies:** Insights from CLTV can guide customer service and loyalty programs to maximize customer satisfaction and retention.

---

## Insights Deep Dive

### Data Preprocessing Steps
To ensure the accuracy of the model, several preprocessing steps were performed:

- **Dropping Negative Values:** Transactions with negative values, often caused by refunds or returns, were removed.
- **Data Type Conversion:** Ensured that columns like `InvoiceDate` were in the correct format.
- **Handling Outliers:** Outliers (extremely high or low values) were capped using the **Interquartile Range (IQR)** method to prevent them from skewing the model's performance.

#### Why Handling Outliers is Important:
Outliers can distort the accuracy of the model by pulling it towards extreme values. By capping outliers, we maintain data integrity without losing valuable information, ensuring more reliable predictions.

### Scaling Features
Feature scaling standardizes the range of variables to make sure all features contribute equally to the model. This is particularly important for algorithms sensitive to the scale of the input data, such as gradient-based methods.

**Benefits:**
- Ensures all features are equally important to the model's learning process.
- Reduces the risk of **numerical instability** during model training.

---

## Model Selection and Performance

### Top Features Selected
Using **SelectKBest** and **f_regression**, the top features that correlated with CLTV were:
- **Number of Transactions**
- **Quantity**
- **Total Revenue**
- **Profit Margin**

Additionally, new columns were created for better model insights:
- **Amount Spent = Price × Quantity**
- **Average Order Value (AOV):** `Total Revenue / Number of Transactions`
- **Customer Lifetime Value (CLTV):** `((AOV * (num_transactions / total_customers)) / (1 - repeat_rate)) * profit_margin`

### Business Summary
- **Total unique customers:** 5,878
- **Repeat Rate:** `98.11%` — A high repeat rate indicates that most customers made more than one transaction, a positive signal for business retention.

### Model Evaluation
We tested several models to predict CLTV and evaluated them using **Mean Squared Error (MSE)**, a metric that measures how close the predicted values are to the actual values. Lower MSE means better model performance.

| Model                        | Cross-Validation MSE |
|-------------------------------|----------------------|
| **Decision Tree**             | 66.6138              |
| **Random Forest**             | 36.7979              |
| **Gradient Boosting**         | 304.7737             |
| **Support Vector Regression** | 18,846,538.6488      |
| **XGBoost**                   | 1,178.5986           |

### Best Model: Random Forest
- **MSE:** 36.7979
- **Why Random Forest?**
  - **Balanced Performance:** It handles outliers and noisy data better than other models.
  - **Business Impact:** A low MSE indicates that this model can accurately predict the value customers bring to the business, allowing for better resource allocation and marketing strategies.

---

## Recommendations
Based on the insights from the analysis, we recommend:
1. **Focus on High-Value Customers:** Use CLTV predictions to create targeted marketing campaigns aimed at retaining top customers.
2. **Optimize Repeat Purchases:** With a high repeat rate (98.11%), invest in loyalty programs and personalized promotions to maintain this trend.
3. **Improve Inventory Planning:** Insights into customer purchasing behavior can help improve stock management and reduce operational costs.
4. **Monitor Low-Value Segments:** Pay attention to customers with lower CLTV predictions and devise strategies to improve their retention.

---

## Further Studies and Limitations
While this analysis provides valuable insights, there are limitations:
1. **Limited Data on Customer Preferences:** The dataset lacks information about customer demographics and preferences, which could provide deeper insights into their behavior.
2. **Seasonality:** We did not account for seasonal trends that might affect purchasing patterns.
3. **External Factors:** Economic conditions, competition, or marketing efforts that might influence CLTV predictions are not considered in this model.

### Future Studies:
- **Incorporate Customer Demographics:** Adding data on customer age, gender, or location could refine the CLTV predictions.
- **Seasonality Effects:** Incorporating time-based variables to account for fluctuations in purchasing behavior due to seasons or holidays.

---
