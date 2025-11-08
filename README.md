This project estimates the selling price of a house based on key features such as size, location, number of bedrooms/bathrooms, and year built.
It uses machine learning regression models trained on the King County House Sales Dataset (Kaggle) to predict prices accurately and interactively via a Streamlit web application.
Data Preprocessing – Cleans and prepares the dataset (handles missing data, encodes categorical variables, scales numerics).
Exploratory Data Analysis (EDA) – Visualizes distributions, correlations, and outliers.
Model Training – Compares multiple regression algorithms:
Linear Regression
Decision Tree Regressor
Random Forest Regressor
XGBoost
LightGBM
CatBoost
Gradient Boosting
Model Evaluation – Uses metrics such as:
RMSE (Root Mean Squared Error)
MAE (Mean Absolute Error)
R² (Coefficient of Determination)
Feature Importance Analysis – Identifies the most influential predictors of price.
Streamlit App – Interactive interface for:
Sinle house price prediction
Batch prediction from CSV upload
Python 3.9+

Libraries: pandas, numpy, scikit-learn, xgboost, lightgbm, catboost, matplotlib, seaborn, joblib

Web Framework: Streamlit
