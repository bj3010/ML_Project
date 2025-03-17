import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
from datetime import date, timedelta
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
import xgboost as xgb
import numpy as np
import warnings

warnings.filterwarnings('ignore')

# API Key
API_KEY = "95b6a65708544e78aa4d63f367a549ba"

# Set Page Config
st.set_page_config(page_title="Stock Market Forecast", layout="wide")

# App Header
st.markdown("# 📈 Stock Market Forecast")
st.markdown("### Analyze & Predict Stock Prices with AI Models and Technical Indicators")
st.divider()

# Sidebar
st.sidebar.header("📌 Select Stock & Date Range")
ticker_list = ["AAPL", "MSFT", "GOOG", "AMZN", "TSLA", "NVDA", "NFLX", "AMD", "INTC", "IBM"]
ticker = st.sidebar.selectbox("🔍 Search & Select Stock", ticker_list, index=0)

st.sidebar.subheader("📅 Select Date Range")
start_date = st.sidebar.date_input("Start Date", value=date.today() - timedelta(days=90))
end_date = st.sidebar.date_input("End Date", value=date.today())
if start_date > end_date:
    st.sidebar.error("⚠️ Start Date cannot be after End Date.")

# Fetch Data
url = f"https://api.twelvedata.com/time_series?symbol={ticker}&interval=1day&apikey={API_KEY}&outputsize=5000"
with st.spinner("Fetching stock data..."):
    try:
        response = requests.get(url)
        data = response.json()
        if "values" not in data:
            st.error(f"⚠️ Error fetching data for {ticker}.")
        else:
            df = pd.DataFrame(data["values"])
            df["datetime"] = pd.to_datetime(df["datetime"], errors='coerce')
            df = df.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"})
            df = df.sort_values(by="datetime")
            df = df[(df["datetime"] >= pd.to_datetime(start_date)) & (df["datetime"] <= pd.to_datetime(end_date))]
            df[["Open", "High", "Low", "Close", "Volume"]] = df[["Open", "High", "Low", "Close", "Volume"]].apply(pd.to_numeric)
            df.insert(0, "Date", df["datetime"], True)

            # Display Data Table
            st.markdown("### 📊 Stock Data Table")
            st.dataframe(df)

            # Stock Price Chart
            st.markdown("### 📈 Stock Price")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], mode='lines', name='Close Price', line=dict(color='blue')))
            st.plotly_chart(fig)

            # Forecasting
            st.markdown("### 🔮 Forecasting with AI Models")
            forecast_models = ["ARIMA", "SARIMA", "Linear Regression", "Random Forest", "Gradient Boosting", "XGBoost", "LSTM", "SVR"]
            selected_model = st.selectbox("Select a forecasting model", forecast_models)
            forecast_period = 3

            def forecast_price(model_name, df, forecast_period):
                y = df["Close"].values.reshape(-1, 1)
                x = pd.Series(range(len(y))).values.reshape(-1, 1)
                
                if model_name == "ARIMA":
                    model = ARIMA(y, order=(5,1,0))
                    model_fit = model.fit()
                    preds = model_fit.forecast(steps=forecast_period)
                elif model_name == "SARIMA":
                    model = SARIMAX(y, order=(1,1,1), seasonal_order=(1,1,1,12))
                    model_fit = model.fit()
                    preds = model_fit.forecast(steps=forecast_period)
                elif model_name == "Linear Regression":
                    model = LinearRegression()
                    model.fit(x, y)
                    preds = model.predict([[len(y) + i] for i in range(1, forecast_period + 1)])
                elif model_name == "Random Forest":
                    model = RandomForestRegressor(n_estimators=100)
                    model.fit(x, y.ravel())
                    preds = model.predict([[len(y) + i] for i in range(1, forecast_period + 1)])
                elif model_name == "Gradient Boosting":
                    model = GradientBoostingRegressor(n_estimators=100)
                    model.fit(x, y.ravel())
                    preds = model.predict([[len(y) + i] for i in range(1, forecast_period + 1)])
                elif model_name == "XGBoost":
                    model = xgb.XGBRegressor(n_estimators=100)
                    model.fit(x, y.ravel())
                    preds = model.predict([[len(y) + i] for i in range(1, forecast_period + 1)])
                elif model_name == "SVR":
                    model = SVR(kernel='rbf')
                    model.fit(x, y.ravel())
                    preds = model.predict([[len(y) + i] for i in range(1, forecast_period + 1)])
                return preds
            
            predictions = forecast_price(selected_model, df, forecast_period)
            prediction_dates = pd.date_range(start=end_date, periods=len(predictions), freq="D")
            prediction_df = pd.DataFrame({"Date": prediction_dates, "Predicted": predictions})
            
            # Forecast Plot
            fig_forecast = go.Figure()
            fig_forecast.add_trace(go.Scatter(x=df["Date"], y=df["Close"], mode='lines', name='Actual', line=dict(color='blue')))
            fig_forecast.add_trace(go.Scatter(x=prediction_df["Date"], y=prediction_df["Predicted"], mode='lines', name='Predicted', line=dict(color='red')))
            fig_forecast.update_layout(title=f'Forecast using {selected_model}', xaxis_title='Date', yaxis_title='Price', template="plotly_dark")
            st.plotly_chart(fig_forecast)
    except Exception as e:
        st.error(f"⚠️ An error occurred: {e}")