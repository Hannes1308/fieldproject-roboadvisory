import streamlit as st
import pandas as pd
import yfinance as yfin
from datetime import datetime
from Green_Robo_Advisor_Class import RoboAdvisor

# Streamlit Page Setup
st.set_page_config(page_title="Green Robo Advisor", layout="wide")
st.title("🌿 Green Robo Advisor")

# Load ETF universe from Excel
try:
    ticker_data = pd.read_excel("Green_ETF_Selection.xlsx", sheet_name="ETF_Universe")
except FileNotFoundError:
    st.error("❌ 'Green_ETF_Selection.xlsx' not found. Please upload or check the file.")
    st.stop()

# Input form
with st.form("user_input_form"):
    name = st.text_input("👤 Your Name", "John Doe")
    risk_level = st.selectbox("📊 Select Risk Profile", ["Low", "Medium", "High"])
    horizon = st.number_input("⏳ Investment Horizon (years)", min_value=1, max_value=50, value=5)
    start_date = st.date_input("📅 Start Date", datetime(2021, 4, 1))
    end_date = st.date_input("📅 End Date", datetime(2025, 4, 1))
    submitted = st.form_submit_button("🚀 Run Optimization")

if submitted:
    try:
        tickers = list(ticker_data.Ticker)
        labels = list(ticker_data.Label)

        st.info("📥 Downloading historical price data...")
        panel_data = yfin.download(tickers, start=start_date, end=end_date)["Close"]
        df = pd.DataFrame({label: panel_data[t] for t, label in zip(ticker_data.Ticker, ticker_data.Label)})
        df = df.resample('D').mean().bfill().ffill()

        # Run Robo Advisor
        st.success("✅ Data loaded successfully. Running optimization...")
        robo = RoboAdvisor(df, rf='Green Bonds', benchmark='MSCI World SRI')

        strategy_map = {
            'Low': 'min-var',
            'Medium': 'max-sharpe-ratio',
            'High': 'max-exp'
        }
        strategy = strategy_map.get(risk_level, 'min-var')
        weights = robo.optimizeWeights(strategy=strategy)

        mu = round(weights @ robo.mu.T * 100, 2)
        sigma = round((weights @ robo.cov @ weights.T)**0.5 * 100, 2)
        sharpe = round((mu / 100 - robo.rf) / (sigma / 100), 2) if sigma != 0 else 0
        weight_dict = {col: round(w * 100, 2) for col, w in zip(df.columns, weights)}

        # Display results
        st.subheader(f"📊 Portfolio Results for {name}")
        col1, col2, col3 = st.columns(3)
        col1.metric("Expected Return (%)", mu)
        col2.metric("Volatility (%)", sigma)
        col3.metric("Sharpe Ratio", sharpe)

        st.subheader("📦 Portfolio Allocation")
        st.bar_chart(pd.Series(weight_dict))

        st.dataframe(pd.DataFrame.from_dict(weight_dict, orient="index", columns=["Weight (%)"]))

    except Exception as e:
        st.error(f"An error occurred during processing: {e}")

