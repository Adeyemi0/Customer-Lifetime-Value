# Customer Lifetime Value Prediction: Data Analysis and Model Selection

## Table of Contents
1. [Background Overview](#background-overview)
2. [Data Dictionary](#data-dictionary)
3. [Executive Summary](#executive-summary)
4. [Insights Deep Dive](#insights-deep-dive)
5. [Recommendations](#recommendations)
6. [Further Studies and Limitations](#further-studies-and-limitations)

---

## Background Overview
This project focuses on predicting **Customer Lifetime Value (CLTV)** by analyzing historical purchase data. CLTV is a crucial metric for understanding customer profitability and guiding marketing and retention strategies. The project involves data preprocessing, feature selection, and building machine learning models to predict CLTV.

---

## Data Dictionary
| **Column**       | **Description**                                   |
|------------------|---------------------------------------------------|
| **Invoice**      | Unique identifier for each transaction             |
| **StockCode**    | Unique product code                                |
| **Description**  | Product description                                |
| **Quantity**     | Number of units purchased                          |
| **InvoiceDate**  | Date and time the transaction occurred             |
| **Price**        | Unit price of the product                          |
| **Customer ID**  | Unique identifier for each customer                |
| **Country**      | Country of the customer                            |

---

## Executive Summary
This analysis aims to identify high-value customers by predicting their CLTV using several models. After preprocessing the data and selecting key features such as the number of transactions, quantity purchased, total revenue, and profit margin, we tested five machine learning models. The **Random Forest** model emerged as the best performer, with a Cross-Validation Mean Squared Error (MSE) of **36.7979**.

### Key Metrics:
- **Number of Unique Customers**: 5,878
- **Repeat Rate**: 98.11% (high customer retention)
- **Best Model**: Random Forest with MSE of 36.7979

These insights help the business focus on retaining and nurturing existing customers, who have a high likelihood of repeating purchases.

---

## Data Preprocessing Steps
### 1. Dropping Negative Values
To maintain data accuracy, negative values in fields like `Quantity` or `Price` were dropped, as they indicate potential errors or returns.

### 2. Handling Outliers
We capped outliers using the **Interquartile Range (IQR)** method:
- **Business Impact**: Reducing the influence of extreme values ensures the model focuses on typical customer behavior rather than anomalies.

### 3. Scaling Features
We standardized feature scales to improve the performance of algorithms:
- **Business Impact**: Ensures no single feature (like revenue or quantity) dominates the predictions.

### Feature Selection: 
We used **SelectKBest** with `f_regression` to identify the most influential features:
- **Selected Features**: Number of transactions, quantity, total revenue, profit margin.

---

## Insights Deep Dive
### Calculated Metrics:
- **Amount Spent** = `price * quantity`
- **Average Order Value (AOV)** = `Total Revenue / Number of Transactions`
- **Customer Lifetime Value (CLTV)** = `((AOV * (num_transactions / total_customers)) / (1 - repeat_rate)) * profit_margin`

These metrics provide insight into customer behavior:
- **AOV** measures how much a customer typically spends in a single transaction.
- **CLTV** predicts the overall value a customer brings during their entire relationship with the business.

---

## Recommendations
Based on the analysis, the business should:
1. **Focus on Customer Retention**: The high repeat rate (98.11%) indicates that customers are likely to return. By focusing on personalized retention strategies, the business can maximize CLTV.
2. **Improve High-Value Customer Targeting**: Using insights from the Random Forest model, the business can allocate more resources to high-value customers with high predicted CLTV.
3. **Optimize Promotions**: Leverage the understanding of customer behavior to design targeted promotions that increase AOV and the number of transactions.

---

## Further Studies and Limitations
### Future Studies:
- **Deep Dive into Customer Segmentation**: Further segment customers by location, purchasing patterns, and product preferences to refine retention strategies.
- **Incorporate Time-Based Features**: Including time-based factors like seasonal effects could improve predictions.

### Limitations:
- **Model Complexity**: While Random Forest performed best, simpler models could be easier to interpret for stakeholders.
- **Data Quality**: Missing data for certain customers or products could affect the accuracy of CLTV predictions.

---
