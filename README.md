# 📈 Crypto Price Prediction App

## 🔥 Overview
This is a **Streamlit-based web application** that predicts cryptocurrency prices using **two machine learning models**:  
- **Prophet** (Time Series Forecasting)
- **Random Forest Regressor** (Supervised Learning)

The app allows users to:
1. **Predict the future price** of a selected cryptocurrency for a specified date between **April 2025 and April 2026**.
2. **Visualize historical and predicted trends** using interactive charts.
3. **Explore prediction charts** for all supported cryptocurrencies and models.

---

## 🏗️ Tech Stack
- **Python**
- **Streamlit** (for the web interface)
- **Pandas** (for data handling)
- **Prophet** (for time series forecasting)
- **Scikit-Learn** (for the Random Forest model)
- **Matplotlib** (for plotting)
- **NumPy** (for numerical operations)

---

## 📊 Machine Learning Models Used

### **1️⃣Prophet**
**Prophet** is an open-source time series forecasting model developed by Facebook.  
✔️ Designed for predicting trends in **financial markets, sales, and demand forecasting**.  
✔️ Handles **seasonality, holidays, and trends** automatically.  
✔️ Uses an **additive model** where trends, seasonality, and holidays are combined.  

🔹 **How it Works?**
- Converts the **date** into a time-series format (`ds` column).
- Uses historical price data to learn trends.
- Generates **future predictions** up to **April 2026**.

🔹 **Advantages:**
- Works well with long time-series datasets.
- Can capture trends and seasonal effects.
- Requires **minimal tuning**.

---

### **2️⃣ Random Forest Regressor**
**Random Forest** is an ensemble learning model used for regression.  
✔️ Built from multiple **decision trees** that collectively predict prices.  
✔️ Uses **historical features like year, month, day, and volume**.  
✔️ Works well with structured tabular data.

🔹 **How it Works?**
- Extracts **features** (date-based properties like month, weekday, quarter, etc.).
- Trains multiple **decision trees** and averages the results.
- Predicts prices **for specific future dates**.

🔹 **Advantages:**
- Works well even with **small datasets**.
- Can model **complex relationships** between variables.
- Less prone to **overfitting** compared to a single decision tree.

---

## 🎮 Features
### 🔮 **Predict**
- Select a **cryptocurrency** (BTC, ETH, SOL, XRP).
- Choose a **prediction model** (Prophet or Random Forest).
- Pick a **future date** (between **April 2025 - April 2026**).
- **Get the predicted price** for the selected date.
- View the **prediction trend** for the crypto.

### 📊 **Data**
- Explore **historical and predicted price trends**.
- View **charts for all cryptocurrencies** using **both models**.
- Analyze **zoomed-in price trends**.

---

## 🚀 Installation & Setup

### **1️⃣ Prerequisites**
Make sure you have the following installed:
- **Python 3.8+**
- **Pip**
- **Git** (optional for cloning)

### **2️⃣ Clone the Repository**
```bash
git clone https://github.com/yourusername/crypto-price-prediction.git
cd crypto-price-prediction
