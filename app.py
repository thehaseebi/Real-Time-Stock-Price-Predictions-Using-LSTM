import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from streamlit_autorefresh import st_autorefresh
from data import fetch_stock_data, fetch_closing_and_premarket, get_last_trading_day
from models import predict_with_lstm
from visualizations import plot_stock
import yfinance as yf

st.set_page_config(layout="wide")

# Custom CSS
st.markdown("""
    <style>
    .price-summary {
        padding: 10px;
        font-size: 28px;
    }
    .prev-close {
        font-size: 14px;
        padding: 5px 10px;
    }
    .chart-container {
        padding: 10px;
    }
    .right-sidebar {
        padding: 10px;
        background-color: #f1f3f5;
    }
    .metric-item {
        margin-bottom: 10px;
        font-size: 14px;
    }
    .compare-to {
        padding: 10px;
        font-size: 12px;
        color: #007bff;
        cursor: pointer;
    }
    .prediction-section {
        padding: 10px;
        background-color: #f8f9fa;
        border-radius: 5px;
        margin-top: 20px;
    }
    .prediction-section h3 {
        font-size: 20px;
        margin-bottom: 10px;
    }
    .prediction-section p {
        font-size: 16px;
        color: #333;
    }
    .refresh-button {
        display: flex;
        align-items: center;
        gap: 5px;
    }
    /* Responsive Design */
    @media (max-width: 1200px) {
        .right-sidebar {
            display: none;
        }
        .chart-container {
            width: 100%;
        }
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar for Inputs
st.sidebar.header("Stock Analysis")

# Fetch S&P 500 tickers and company names from Wikipedia
try:
    sp500_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    df = pd.read_html(sp500_url)[0]
    ticker_to_company = dict(zip(df["Symbol"], df["Security"]))
    company_names = list(ticker_to_company.values())
    tickers = list(ticker_to_company.keys())
    print(f"Fetched {len(tickers)} tickers and company names from S&P 500 list")
except Exception as e:
    st.error(f"Error fetching S&P 500 tickers: {str(e)}")
    ticker_to_company = {"NVDA": "NVIDIA Corp", "AAPL": "Apple Inc", "MSFT": "Microsoft Corp", "GOOGL": "Alphabet Inc",
                         "AMZN": "Amazon.com Inc"}
    company_names = list(ticker_to_company.values())
    tickers = list(ticker_to_company.keys())

# Create dropdown with company names
selected_company = st.sidebar.selectbox("Select Company", company_names, index=company_names.index(
    "NVIDIA Corp") if "NVIDIA Corp" in company_names else 0)
ticker = tickers[company_names.index(selected_company)]

# Prediction Inputs
st.sidebar.header("Stock Price Prediction")
try:
    default_future_date = datetime.now().date() + timedelta(days=1)
    future_date = st.sidebar.date_input("Select Future Date for Prediction",
                                       value=default_future_date,
                                       min_value=datetime.now().date())
except Exception as e:
    st.sidebar.error(f"Error setting future date: {str(e)}")
    future_date = datetime.now().date() + timedelta(days=1)

predict_button = st.sidebar.button("Predict Stock Price")

# Refresh Controls
st.sidebar.header("Data Refresh")
if st.sidebar.button("🔄", key="manual_refresh"):
    st.rerun()

# Auto-refresh toggle
auto_refresh = st.sidebar.checkbox("Enable Auto-Refresh (every 5 minutes)", value=False)
if auto_refresh:
    st_autorefresh(interval=300000, key="datarefresh")  # 5 minutes in milliseconds

# Fetch Stock Info
try:
    stock_info = fetch_closing_and_premarket(ticker)
    print("Stock Info:", stock_info)
except Exception as e:
    st.error(f"Error fetching stock info: {str(e)}")
    stock_info = None

# Main Layout with Two Columns
col_left, col_right = st.columns([3, 1])

# Left Column: Price Summary, Chart, and Prediction
with col_left:
    # Price Summary (including company name)
    if stock_info:
        st.markdown(f"""
            <div style='font-size: 24px; font-weight: bold; padding-bottom: 10px;'>
                {stock_info['company_name']} ({ticker.upper()}) - {stock_info['exchange']}
            </div>
            <div class="price-summary">
                ${stock_info['current_price']:.2f} 
                <span style='color: {"green" if stock_info['percent_change'] >= 0 else "red"};'>
                    {'↑' if stock_info['percent_change'] >= 0 else '↓'}{stock_info['percent_change']:.2f}% 
                    {'+' if stock_info['price_change'] >= 0 else ''}{stock_info['price_change']:.2f} Today
                </span>
            </div>
            <div class="prev-close">
                Previous Closed Market: ${stock_info['previous_close']:.2f} 
                <span style='color: {"green" if stock_info['prev_close_percent_change'] >= 0 else "red"};'>
                    ({'↑' if stock_info['prev_close_percent_change'] >= 0 else '↓'}{stock_info['prev_close_percent_change']:.2f}%) 
                    {'+' if stock_info['prev_close_change'] >= 0 else ''}{stock_info['prev_close_change']:.2f}
                </span>
                <br>Closed: {stock_info['closing_time']} - USD {stock_info['exchange']} - Disclaimer
            </div>
        """, unsafe_allow_html=True)

    # Fetch and Display Chart Data (Live Mode)
    df = None
    predicted_price = None  # Initialize predicted_price
    try:
        # Try fetching interval data for the current day
        start_date = datetime(2015, 1, 1, 0, 0, 0)
        end_date = datetime.now()
        interval = "1m"
        df = fetch_stock_data(ticker, start_date, end_date, interval=interval)
        st.write(f"Last Updatedñana: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} (Live Data)")
    except Exception as e:
        # Fallback to data for the last trading day
        #st.warning("Market closed! Showing data for the last trading day.")
        try:
            last_trading_day = get_last_trading_day(ticker)
            if last_trading_day:
                start_date = datetime.combine(last_trading_day, datetime.min.time())
                end_date = datetime.combine(last_trading_day, datetime.max.time())
                interval = "1m"
                df = fetch_stock_data(ticker, start_date, end_date, interval=interval)
                st.write(f"Showing data for: {last_trading_day.strftime('%Y-%m-%d')} (Last Trading Day)")
                print(f"Last trading day used: {last_trading_day}")
            else:
                st.warning("Could not determine the last trading day.")
        except Exception as fallback_e:
            st.error(f"Error fetching data for last trading day: {str(fallback_e)}")

    # Prediction Section
    if predict_button:
        try:
            # Fetch historical data for prediction (last 4 years)
            end_date_pred = datetime.now().date()
            start_date_pred = datetime(2015, 1, 1, 0, 0, 0)  # Use 2015-01-01 as requested
            df_pred = fetch_stock_data(ticker, start_date_pred, end_date_pred, interval="1d")

            if df_pred is None or df_pred.empty:
                st.error("No historical data available for prediction.")
            else:
                predicted_price = predict_with_lstm(df_pred, future_date)

                st.markdown('<div class="prediction-section">', unsafe_allow_html=True)
                st.markdown(f"""
                    <h3>Predicted Stock Price</h3>
                    <p>The predicted price for {ticker} on {future_date} is <strong>${predicted_price:.2f}</strong>.</p>
                """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error predicting stock price: {str(e)}")

    # Display Chart with Predicted Price (if available)
    if df is not None and not df.empty:
        print(f"Fetched DataFrame:\n{df.head()}\n{df.tail()}")
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        # Pass predicted_price and future_date to plot_stock
        st.plotly_chart(
            plot_stock(df, ticker, "Line", predicted_price=predicted_price, future_date=future_date),
            use_container_width=True
        )
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.warning("No data available to display. This may be due to a non-trading period or unavailable data.")

# Right Column: Metrics
with col_right:
    if stock_info:
        st.markdown('<div class="right-sidebar">', unsafe_allow_html=True)
        st.markdown(f"""
            <div class="metric-item"><strong>Previous Close:</strong> ${stock_info['previous_close']:.2f}</div>
            <div class="metric-item"><strong>Day Range:</strong> ${stock_info['day_low']:.2f} - ${stock_info['day_high']:.2f}</div>
            <div class="metric-item"><strong>Year Range:</strong> ${stock_info['year_low']:.2f} - ${stock_info['year_high']:.2f}</div>
            <div class="metric-item"><strong>Market Cap:</strong> {stock_info['market_cap']}</div>
            <div class="metric-item"><strong>Primary Exchange:</strong> {stock_info['exchange']}</div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)