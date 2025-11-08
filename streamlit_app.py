import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ------------------------ Load  model -------------------------
@st.cache_resource  
def load_model():
    return joblib.load("best_house_price_model.pkl")  

model = load_model()

# ------------------------ Web Page  -------------------------
st.set_page_config(page_title="Project: House Price Prediction")
st.title("House Price Prediction App")
st.write("Use this app to estimate the selling price of a house based on its features (size, location, and year built).")

# ------------------------ Tabs: Single vs Batch -------------------------
tab_single, tab_batch = st.tabs(["Single house prediction", "Batch prediction from CSV"])

# ^^^^^^^^^  feature columns ^^^^^^^^^
FEATURE_COLS = [
    "bedrooms",
    "bathrooms",
    "sqft_living",
    "sqft_lot",
    "floors",
    "waterfront",
    "view",
    "condition",
    "grade",
    "sqft_above",
    "sqft_basement",
    "yr_built",
    "yr_renovated",
    "zipcode",
    "lat",
    "long",
    "sqft_living15",
    "sqft_lot15",
    "year_sold",
    "month_sold",
]

# ------------------------ single House ------------------------
with tab_single:
    st.subheader("Single house input")

    col1, col2 = st.columns(2)

    with col1:
        bedrooms = st.number_input("Bedrooms", min_value=0, max_value=15, value=3, step=1)
        bathrooms = st.number_input("Bathrooms", min_value=0.0, max_value=10.0, value=2.0, step=0.25)
        sqft_living = st.number_input("Living area (sqft_living)", min_value=200, max_value=10000, value=1800, step=50)
        sqft_lot = st.number_input("Lot size (sqft_lot)", min_value=400, max_value=100000, value=5000, step=100)
        floors = st.number_input("Floors", min_value=1.0, max_value=4.0, value=1.0, step=0.5)
        waterfront = st.selectbox("Waterfront (0 = No, 1 = Yes)", [0, 1])
        view = st.number_input("View (0-4)", min_value=0, max_value=4, value=0, step=1)
        condition = st.number_input("Condition (1-5)", min_value=1, max_value=5, value=3, step=1)
        grade = st.number_input("Grade (1-13)", min_value=1, max_value=13, value=7, step=1)
        sqft_above = st.number_input("Sqft above ground (sqft_above)", min_value=200, max_value=8000, value=1500, step=50)

    with col2:
        sqft_basement = st.number_input("Basement area (sqft_basement)", min_value=0, max_value=5000, value=300, step=50)
        yr_built = st.number_input("Year built (yr_built)", min_value=1800, max_value=2025, value=1990, step=1)
        yr_renovated = st.number_input("Year renovated (0 if never)", min_value=0, max_value=2025, value=0, step=1)
        zipcode = st.text_input("Zipcode (e.g. 98178)", value="98178")
        lat = st.number_input("Latitude (lat)", min_value=47.0, max_value=48.0, value=47.5, step=0.001, format="%.3f")
        long = st.number_input("Longitude (long)", min_value=-123.5, max_value=-121.0, value=-122.2, step=0.001, format="%.3f")
        sqft_living15 = st.number_input("sqft_living15 (neighbor living area)", min_value=200, max_value=8000, value=1800, step=50)
        sqft_lot15 = st.number_input("sqft_lot15 (neighbor lot size)", min_value=400, max_value=80000, value=5000, step=100)
        year_sold = st.number_input("Year sold (year_sold)", min_value=2010, max_value=2025, value=2014, step=1)
        month_sold = st.number_input("Month sold (month_sold)", min_value=1, max_value=12, value=6, step=1)

    # --------  DataFrame from user input --------
    input_dict = {
        "bedrooms": [bedrooms],
        "bathrooms": [bathrooms],
        "sqft_living": [sqft_living],
        "sqft_lot": [sqft_lot],
        "floors": [floors],
        "waterfront": [waterfront],
        "view": [view],
        "condition": [condition],
        "grade": [grade],
        "sqft_above": [sqft_above],
        "sqft_basement": [sqft_basement],
        "yr_built": [yr_built],
        "yr_renovated": [yr_renovated],
        "zipcode": [zipcode],
        "lat": [lat],
        "long": [long],
        "sqft_living15": [sqft_living15],
        "sqft_lot15": [sqft_lot15],
        "year_sold": [year_sold],
        "month_sold": [month_sold],
    }

    input_df = pd.DataFrame(input_dict)
    st.subheader("Input preview")
    st.dataframe(input_df)

    st.subheader("Prediction result")
    if st.button("Predict house price"):
        pred_price = model.predict(input_df)[0]
        st.success(f"Estimated house price: **${pred_price:,.0f}**")

# ------------------------  Batch prediction ------------------------
with tab_batch:
    st.subheader("Batch prediction from CSV")
    st.write("Upload a CSV file similar to the  `kc_house_data` file ,e app will create `year_sold` and `month_sold` from the `date` column if present,then predict prices for all rows.")

    uploaded_file = st.file_uploader("Upload .CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            df_batch = pd.read_csv(uploaded_file)
            st.write("Preview of uploaded data:")
            st.dataframe(df_batch.head())

            # Convert date ( datetime and create year_sold, month_sold if date exists
            if "date" in df_batch.columns:
                df_batch["date"] = pd.to_datetime(df_batch["date"])
                df_batch["year_sold"] = df_batch["date"].dt.year
                df_batch["month_sold"] = df_batch["date"].dt.month

            # Ensure zipcode is string--------------------
            if "zipcode" in df_batch.columns:
                df_batch["zipcode"] = df_batch["zipcode"].astype(str)

            # Check required feature columns--------------------
            missing = [c for c in FEATURE_COLS if c not in df_batch.columns]
            if missing:
                st.error(f"Missing required columns in CSV: {missing}")
            else:
                X_batch = df_batch[FEATURE_COLS].copy()

                # Predict--------------------
                predictions = model.predict(X_batch)

                results_df = df_batch.copy()
                results_df["Predicted_Price"] = predictions

                st.write("Batch prediction results (first rows):")
                st.dataframe(results_df.head())

                # Download results--------------------
                csv_download = results_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="Download results as CSV",
                    data=csv_download,
                    file_name="house_price_predictions.csv",
                    mime="text/csv",
                )

        except Exception as e:
            st.error(f"Error reading or processing file: {e}")
