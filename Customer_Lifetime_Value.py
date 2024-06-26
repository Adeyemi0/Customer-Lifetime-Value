# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
import xgboost as xgb
from sklearn.metrics import mean_squared_error

# Load the dataset
df = pd.read_csv('online_retail.csv')

# Define the sidebar menu
st.sidebar.title("Menu")
app_mode = st.sidebar.selectbox("Choose the page", ["Data Preprocessing", "CLTV Prediction"])

# Define the Data Preprocessing page
if app_mode == "Data Preprocessing":
    st.title("Data Preprocessing for CLTV Prediction")

    st.write("""
    This section covers the preprocessing and feature extraction steps applied to the `online_retail` dataset before predicting Customer Lifetime Value (CLTV).
    """)

    # Display the first few rows of the dataset
    st.write("### Original Dataset")
    st.write(df.head())

    # Show dataset description
    st.write("### Dataset Description")
    st.write("Below are the summary statistics of the original dataset. Notice the presence of negative values which indicate erroneous data entries that need to be addressed.")
    st.write(df.describe())

    # Handling missing values and negative values
    st.write("### Handling Missing Values and Negative Values")
    df = df.dropna()
    df = df[(df["Price"] > 0) & (df["Quantity"] > 0)]
    st.write("Dataset after dropping rows with missing and negative values:")
    st.write(df.describe())

    # Converting dates
    st.write("### Converting `InvoiceDate` to Datetime Format")
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    st.write("Dataset with `InvoiceDate` converted to datetime format:")
    st.write(df.dtypes)

    # Adding 'Amount Spent'
    st.write("### Adding `Amount Spent` Column")
    df["Amount Spent"] = df["Quantity"] * df["Price"]
    st.write("Dataset with `Amount Spent` column added:")
    st.write(df.head())

    # Aggregating data per customer
    st.write("### Aggregating Data per Customer")

    def customer_model(data):
        max_date = data['InvoiceDate'].max()
        data = data.groupby('Customer ID').agg({
            'InvoiceDate': lambda x: (max_date - x.min()).days,
            'Invoice': lambda x: len(x),
            'Quantity': lambda x: x.sum(),
            'Amount Spent': lambda x: x.sum()
        })
        return data

    data = customer_model(df)
    st.write("Aggregated customer data:")
    st.write(data.head())

    # Renaming columns and calculating additional features
    data.columns = ['customer_loyalty_period (days)', 'num_transactions', 'quantity', 'total_revenue']
    data = data[data['quantity'] > 0]

    # Calculate Average Order Value (AOV)
    st.write("### Calculating Average Order Value (AOV)")
    st.write("""
    The Average Order Value (AOV) is calculated using the formula:
    AOV = Total Revenue / Number of Transactions
    """)
    data['AOV'] = data['total_revenue'] / data['num_transactions']
    st.write("Dataset with `AOV` calculated:")
    st.write(data.head())

    # Calculate Total Customers
    total_customers = len(data)
    st.write("### Total Number of Customers")
    st.write(f"The total number of unique customers is: {total_customers}")

    # Calculate Repeat Rate
    st.write("### Calculating Repeat Rate")
    st.write("""
    The Repeat Rate is calculated using the formula:
    Repeat Rate = Number of Customers with More Than One Transaction / Total Number of Customers
    """)
    repeat_rate = data[data['num_transactions'] > 1].shape[0] / total_customers
    st.write(f"The calculated repeat rate is: {repeat_rate:.2%}")

    # Adding Profit Margin
    data['profit_margin'] = data['total_revenue'] * 0.10

    # Calculating CLTV
    st.write("### Calculating CLTV")
    st.write("""
    Customer Lifetime Value (CLTV) is calculated using the following formula:
    CLTV = ((AOV * (num_transactions / total_customers)) / (1 - repeat_rate)) * profit_margin
    - AOV: Average Order Value
    - num_transactions: Number of transactions per customer
    - total_customers: Total number of customers
    - repeat_rate: The rate at which customers make repeat purchases
    """)
    data['CLTV'] = ((data['AOV'] * (data['num_transactions'] / total_customers)) / (1 - repeat_rate)) * data['profit_margin']
    st.write("Dataset with CLTV calculated:")
    st.write(data.head())

    # Boxplot for original data (extracted features)
    st.write("### Boxplots Before Removing Outliers")
    st.write("The boxplots below show the distribution for the extracted features. Outliers can significantly affect the quality of our model.")
    fig, ax = plt.subplots(2, 2, figsize=(14, 10))
    sns.boxplot(x=data["customer_loyalty_period (days)"], ax=ax[0, 0])
    ax[0, 0].set_title('Customer Loyalty Period')
    sns.boxplot(x=data["num_transactions"], ax=ax[0, 1])
    ax[0, 1].set_title('Number of Transactions')
    sns.boxplot(x=data["total_revenue"], ax=ax[1, 0])
    ax[1, 0].set_title('Total Revenue')
    sns.boxplot(x=data["CLTV"], ax=ax[1, 1])
    ax[1, 1].set_title('CLTV')
    st.pyplot(fig)

    # Technical explanation of outlier removal
    st.write("### Handling Outliers")
    st.write("""
    Outliers are data points that differ significantly from other observations and can adversely affect the model. I handled outliers by capping them at the upper and lower limits based on the interquartile range (IQR). This method is preferred because:

    - **Simplicity:** It’s straightforward and doesn’t require advanced statistical knowledge.
    - **Robustness:** It effectively reduces the impact of extreme values without completely discarding them.
    - **Preservation of Data Integrity:** Instead of removing data points, which could lead to loss of valuable information, we adjust them to a reasonable range.
    """)

    # Removing outliers
    st.write("### Removing Outliers")

    def outlier_thresholds(dataframe, variable):
        quartile1 = dataframe[variable].quantile(0.25)
        quartile3 = dataframe[variable].quantile(0.75)
        interquartile_range = quartile3 - quartile1
        up_limit = quartile3 + 1.5 * interquartile_range
        low_limit = quartile1 - 1.5 * interquartile_range
        return low_limit, up_limit

    def replace_with_thresholds(dataframe, variable):
        low_limit, up_limit = outlier_thresholds(dataframe, variable)
        dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
        dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

    for column in ['customer_loyalty_period (days)', 'num_transactions', 'quantity', 'total_revenue', 'AOV', 'profit_margin', 'CLTV']:
        replace_with_thresholds(data, column)

    st.write("Data after removing outliers:")
    st.write(data.describe())

    # Boxplot for processed data
    st.write("### Boxplots After Removing Outliers")
    st.write("The following boxplots show the distribution for the extracted features after handling outliers.")
    fig, ax = plt.subplots(2, 2, figsize=(14, 10))
    sns.boxplot(x=data["customer_loyalty_period (days)"], ax=ax[0, 0])
    ax[0, 0].set_title('Customer Loyalty Period')
    sns.boxplot(x=data["num_transactions"], ax=ax[0, 1])
    ax[0, 1].set_title('Number of Transactions')
    sns.boxplot(x=data["total_revenue"], ax=ax[1, 0])
    ax[1, 0].set_title('Total Revenue')
    sns.boxplot(x=data["CLTV"], ax=ax[1, 1])
    ax[1, 1].set_title('CLTV')
    st.pyplot(fig)

    # Scaling features
    st.write("### Scaling Features")
    st.write("""
    Feature scaling is performed to standardize the range of independent variables or features of data. The goal is to bring all features to a similar scale so that no single feature dominates the learning process. 

    **Scaling Benefits:**
    - **Consistent Range:** Ensures that each feature contributes equally to the model.
    - **Algorithm Sensitivity:** Improves performance of distance-based and gradient descent algorithms.
    - **Numerical Stability:** Reduces numerical instability and improves convergence.
    """)

    X = data[['customer_loyalty_period (days)', 'num_transactions', 'quantity', 'total_revenue', 'AOV', 'profit_margin']]
    y = data['CLTV']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    st.write("Data after scaling:")
    st.write(pd.DataFrame(X_scaled, columns=['customer_loyalty_period (days)', 'num_transactions', 'quantity', 'total_revenue', 'AOV', 'profit_margin']).head())

    # Selecting top features
    st.write("### Feature Selection")
    st.write("""
    The top features were selected using SelectKBest with f_regression.
    This step helps in identifying the features most correlated with the target variable (CLTV)

    """)
    from sklearn.feature_selection import SelectKBest, f_regression
    selector = SelectKBest(score_func=f_regression, k=4)
    X_selected = selector.fit_transform(X_scaled, y)

    selected_features = X.columns[selector.get_support()]
    st.write("Selected features:", selected_features)

    # Prepare final dataset for modeling
    final_data = data[selected_features]
    final_data['CLTV'] = y

    st.write("### Final Dataset for Modeling")
    st.write(final_data.head())

    # Save final data for use in the modeling page
    final_file = 'final_dataset.csv'
    final_data.to_csv(final_file, index=False)
    st.download_button(
        label="Download Final Dataset",
        data=final_data.to_csv(index=False).encode('utf-8'),
        file_name=final_file,
        mime='text/csv'
    )

# Define the CLTV Prediction page
elif app_mode == "CLTV Prediction":
    st.title("Customer Lifetime Value (CLTV) Prediction")

    st.write("""
    This section uses a pre-trained model to predict the Customer Lifetime Value (CLTV) based on features derived from the `online_retail` dataset.
    """)

    # Load final dataset
    try:
        final_data = pd.read_csv('final_dataset.csv')
    except Exception as e:
        st.error(f"Error loading final dataset: {e}")
        final_data = None

    if final_data is not None:
        st.write("### Final Dataset Used for Modeling")
        st.write(final_data.head())

        X_final = final_data.drop(columns=['CLTV'])
        y_final = final_data['CLTV']

        # Select the best model for prediction
        st.write("### Selecting the Best Model")

        # Splitting the dataset
        X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=0.2, random_state=42)

        # Initialize models
        models = {
            'Decision Tree': DecisionTreeRegressor(random_state=42),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'Support Vector Regression': SVR(),
            'XGBoost': xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
        }

        mse_results = {}
        for name, model in models.items():
            cv_scores = -cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
            mse = np.mean(cv_scores)
            mse_results[name] = mse
            st.write(f'{name} Cross-Validation MSE: {mse:.4f}')

        best_model_name = min(mse_results, key=mse_results.get)
        st.write(f'\n**Best Model:** {best_model_name} with Cross-Validation MSE: {mse_results[best_model_name]:.4f}')

        best_model = models[best_model_name]
        best_model.fit(X_train, y_train)

        # Save the best model
        model_file = 'best_CLTV_model.pkl'
        with open(model_file, 'wb') as file:
            pickle.dump(best_model, file)
        st.write(f'Saved the best model as `{model_file}`.')

        st.write("### Testing the Model with New Dataset")

        st.write("""
        Upload a new CSV file containing the following columns in the same order:
        - `num_transactions`
        - `quantity`
        - `total_revenue`
        - `profit_margin`
        """)

        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

        if uploaded_file:
            new_data = pd.read_csv(uploaded_file)
            st.write("Uploaded Data Preview:")
            st.write(new_data.head())

            if list(new_data.columns) == list(X_final.columns):
                try:
                    with open(model_file, 'rb') as file:
                        loaded_model = pickle.load(file)

                    if hasattr(loaded_model, 'predict'):
                        predictions = loaded_model.predict(new_data)
                        new_data['Predicted_CLTV'] = predictions
                        st.write("Predictions:")
                        st.write(new_data)

                        st.download_button(
                            label="Download predictions as CSV",
                            data=new_data.to_csv(index=False).encode('utf-8'),
                            file_name='predicted_cltv.csv',
                            mime='text/csv'
                        )
                    else:
                        st.error("Loaded model is not a valid machine learning model.")
                except Exception as e:
                    st.error(f"Error predicting with the model: {e}")
            else:
                st.error("The uploaded dataset does not match the required format. Please check the column names and order.")

    else:
        st.error("Final dataset is not available. Please preprocess the data first.")
