import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from prophet import Prophet

# Load Data
@st.cache_data
def load_data():
    df = pd.read_csv('Crypto.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    df.sort_values(by='Date', inplace=True)
    return df

crypto_df = load_data()

# Sidebar Navigation
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio("Go to", ["🔮 Predict", "📊 Data"])

# --- PREDICTION PAGE ---
if page == "🔮 Predict":
    st.title("🔮 Crypto Price Prediction")

    # Select Crypto and Model
    cryptos = crypto_df['Crypto'].unique().tolist()
    selected_crypto = st.selectbox("Choose a cryptocurrency", cryptos)
    selected_model = st.radio("Choose a prediction model", ['Prophet', 'Random Forest'])

    # Select Future Date (from April 2025 to April 2026)
    future_dates = pd.date_range(start="2025-04-01", end="2026-04-01", freq='MS')
    selected_future_date = st.selectbox("Choose a future date", future_dates.date)

    # Filter Data for Selected Crypto
    df = crypto_df[crypto_df['Crypto'] == selected_crypto].copy()
    
    # Prediction Variables
    predicted_price = None  
    forecast_df = None  

    # --- PROPHET MODEL ---
    if selected_model == 'Prophet':
        prophet_df = df[['Date', 'Price']].rename(columns={'Date': 'ds', 'Price': 'y'})
        model_prophet = Prophet()
        model_prophet.fit(prophet_df)

        future = model_prophet.make_future_dataframe(periods=(future_dates[-1] - df['Date'].max()).days)
        forecast = model_prophet.predict(future)

        predicted_price = forecast[forecast['ds'].dt.date == selected_future_date]['yhat'].values
        forecast_df = forecast[['ds', 'yhat']]

    # --- RANDOM FOREST MODEL ---
    elif selected_model == 'Random Forest':
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['Day'] = df['Date'].dt.day
        df['Weekday'] = df['Date'].dt.weekday
        df['Quarter'] = df['Date'].dt.quarter
        df['DayOfYear'] = df['Date'].dt.dayofyear

        if 'Vol.' in df.columns:
            df['Vol.'] = pd.to_numeric(df['Vol.'], errors='coerce').fillna(df['Vol.'].median())
            features = ['Vol.', 'Year', 'Month', 'Day', 'Weekday', 'Quarter', 'DayOfYear']
        else:
            features = ['Year', 'Month', 'Day', 'Weekday', 'Quarter', 'DayOfYear']

        rf = RandomForestRegressor(n_estimators=1000, max_depth=20, min_samples_split=3, min_samples_leaf=1, random_state=42)
        rf.fit(df[features], df['Price'])

        future_df = pd.DataFrame({'Date': future_dates})
        future_df['Year'] = future_df['Date'].dt.year
        future_df['Month'] = future_df['Date'].dt.month
        future_df['Day'] = future_df['Date'].dt.day
        future_df['Weekday'] = future_df['Date'].dt.weekday
        future_df['Quarter'] = future_df['Date'].dt.quarter
        future_df['DayOfYear'] = future_df['Date'].dt.dayofyear
        if 'Vol.' in df.columns:
            future_df['Vol.'] = df['Vol.'].median()

        future_predictions = rf.predict(future_df[features])
        forecast_df = pd.DataFrame({'ds': future_df['Date'], 'yhat': future_predictions})
        predicted_price = forecast_df[forecast_df['ds'].dt.date == selected_future_date]['yhat'].values

    # Display Predicted Price
    if predicted_price is not None and len(predicted_price) > 0:
        st.success(f"📅 Predicted price for {selected_crypto} on {selected_future_date}: **{predicted_price[0]:,.2f} USD**")
    else:
        st.warning("No prediction available for this date.")

    # Plot General Trend
    if forecast_df is not None:
        st.subheader(f"📊 General Trend for {selected_crypto} with {selected_model}")

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df['Date'], df['Price'], label='Actual Price', color='blue')
        ax.plot(forecast_df['ds'], forecast_df['yhat'], label='Predicted', color='green', linestyle='dashed')

        if predicted_price is not None and len(predicted_price) > 0:
            ax.scatter(pd.Timestamp(selected_future_date), predicted_price[0], color='red', label='Selected Value', s=100)

        ax.set_title(f"Price Evolution with {selected_model} for {selected_crypto}")
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

# --- DATA PAGE ---
elif page == "📊 Data":
    st.title("📊 Historical Data & Predictions")

    # Display charts for all cryptos
    for crypto in crypto_df['Crypto'].unique():
        st.subheader(f"📈 {crypto} Predictions")

        # Prophet
        model_prophet = Prophet()
        prophet_df = crypto_df[crypto_df['Crypto'] == crypto][['Date', 'Price']].rename(columns={'Date': 'ds', 'Price': 'y'})
        model_prophet.fit(prophet_df)
        future = model_prophet.make_future_dataframe(periods=365)
        forecast = model_prophet.predict(future)

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(prophet_df['ds'], prophet_df['y'], label='Actual', color='blue')
        ax.plot(forecast['ds'], forecast['yhat'], label='Prophet Prediction', color='green', linestyle='dashed')
        ax.set_title(f"{crypto} - Prophet Model")
        ax.legend()
        st.pyplot(fig)

        # Random Forest (not trained separately for all cryptos, but can be added if needed)
