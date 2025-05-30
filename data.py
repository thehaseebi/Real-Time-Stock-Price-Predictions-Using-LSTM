# data.py
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import streamlit as st

@st.cache_data
def fetch_stock_data(ticker, start_date, end_date, interval="1d"):
    """Fetch historical stock data from Yahoo Finance with specified interval."""
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date, end=end_date, interval=interval)
        if df.empty:
            raise ValueError(
                f"No data found for ticker {ticker} from {start_date} to {end_date} with interval {interval}. "
                f"This may be due to non-trading hours, weekends, holidays, or unavailable data.")
        print(f"Fetched data for {ticker} from {start_date} to {end_date}, interval {interval}. "
              f"Data range: {df.index[0].date()} to {df.index[-1].date()}")
        return df[['Open', 'High', 'Low', 'Close', 'Volume']]
    except Exception as e:
        raise Exception(f"Error fetching data: {str(e)}")

@st.cache_data
def fetch_realtime_data(ticker):
    """Fetch live data for the last 5 days."""
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period="5d", interval="1m")
        if df.empty:
            raise ValueError(f"No data found for ticker {ticker}. This may be due to non-trading periods or unavailable data.")
        print(f"Fetched realtime data for {ticker}. "
              f"Data range: {df.index[0].date()} to {df.index[-1].date()}")
        return df[['Open', 'High', 'Low', 'Close', 'Volume']]
    except Exception as e:
        raise Exception(f"Error fetching data: {str(e)}")


def fetch_closing_and_premarket(ticker):
    """Fetch the most recent closing price, previous close metrics, company name, and additional metrics."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        history = stock.history(period="3d")  # Fetch 3 days to compute previous close changes

        # Debug: Print available info keys
        print("Available info keys:", info.keys())

        # Company Name
        company_name = info.get('longName', 'Unknown Company')
        if company_name == 'Unknown Company':
            print(f"Warning: 'longName' not found for {ticker}, using fallback")

        exchange = info.get('exchange', 'N/A')

        # Current and Previous Close
        current_price = info.get('regularMarketPrice', None)
        previous_close = info.get('previousClose', None)
        if previous_close is None or history.empty:
            raise ValueError(f"Unable to fetch closing price for {ticker}")

        # Compute Previous Close Metrics (compare previous close to day before that)
        if len(history) >= 2:
            prev_prev_close = history['Close'].iloc[-3] if len(history) >= 3 else history['Close'].iloc[-2]
            prev_close_change = previous_close - prev_prev_close
            prev_close_percent_change = (prev_close_change / prev_prev_close) * 100 if prev_prev_close != 0 else 0
        else:
            prev_close_change = 0
            prev_close_percent_change = 0

        # Closing Time
        closing_time = history.index[-1].strftime('%B %d at %I:%M:%S %p %Z') if not history.empty else "Unknown"

        # Other Metrics
        day_low = info.get('regularMarketDayLow', 'N/A')
        day_high = info.get('regularMarketDayHigh', 'N/A')
        year_low = info.get('fiftyTwoWeekLow', 'N/A')
        year_high = info.get('fiftyTwoWeekHigh', 'N/A')
        market_cap = info.get('marketCap', 'N/A')
        if market_cap != 'N/A':
            market_cap = f"{market_cap / 1e12:.2f}T" if market_cap > 1e12 else f"{market_cap / 1e9:.2f}B"
        avg_volume = info.get('averageVolume', 'N/A')
        if avg_volume != 'N/A':
            avg_volume = f"{avg_volume / 1e6:.2f}M" if avg_volume > 1e6 else f"{avg_volume / 1e3:.2f}K"
        pe_ratio = info.get('trailingPE', 'N/A')
        if pe_ratio != 'N/A':
            pe_ratio = f"{pe_ratio:.2f}"
        dividend_yield = info.get('dividendYield', 'N/A')
        if dividend_yield != 'N/A':
            dividend_yield = f"{dividend_yield * 100:.2f}%"

        # Compute Current Day Changes (for top-level price summary)
        price_change = current_price - previous_close if current_price is not None else 0
        percent_change = (price_change / previous_close) * 100 if previous_close != 0 else 0

        return {
            'company_name': company_name,
            'exchange': exchange,
            'current_price': current_price if current_price is not None else previous_close,
            'price_change': price_change,
            'percent_change': percent_change,
            'previous_close': previous_close,
            'prev_close_change': prev_close_change,
            'prev_close_percent_change': prev_close_percent_change,
            'closing_time': closing_time,
            'day_low': day_low,
            'day_high': day_high,
            'year_low': year_low,
            'year_high': year_high,
            'market_cap': market_cap,
            'avg_volume': avg_volume,
            'pe_ratio': pe_ratio,
            'dividend_yield': dividend_yield
        }
    except Exception as e:
        raise Exception(f"Error fetching closing/previous close data: {str(e)}")


def get_default_date_range():
    """Return default start and end dates (1 day)."""
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=1)
    return start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')


def get_last_trading_day(ticker):
    """Determine the most recent trading day for the given ticker."""
    try:
        stock = yf.Ticker(ticker)
        # Fetch recent daily data to find the last trading day
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=7)  # Look back up to 7 days
        df = stock.history(start=start_date, end=end_date, interval="1d")
        if df.empty:
            print(f"No trading days found for {ticker} in the last 7 days.")
            return None
        last_trading_day = df.index[-1].date()
        print(f"Detected last trading day for {ticker}: {last_trading_day}")
        return last_trading_day
    except Exception as e:
        print(f"Error finding last trading day for {ticker}: {str(e)}")
        return None