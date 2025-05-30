
# Stock Price Prediction Dashboard

This project provides a real-time stock analysis and forecasting dashboard using Streamlit, Yahoo Finance API, and a Long Short-Term Memory (LSTM) neural network model. It enables users to visualize historical and live stock market data and predict future stock prices for S&P 500 companies.

## Features

- Live stock data retrieval from Yahoo Finance
- Interactive price charts using Plotly
- LSTM-based future price prediction
- Technical indicators: SMA, EMA, RSI, volatility, lagged values
- Custom future date selection for forecasting
- Data auto-refresh functionality (optional every 5 minutes)
- Sidebar interface for selecting companies and parameters

## Project Structure

```
.
├── app.py               # Streamlit interface and dashboard layout
├── data.py              # Data extraction and processing from Yahoo Finance
├── models.py            # LSTM model creation and prediction logic
├── visualizations.py    # Plotly-based charting utilities
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/thehaseebi/Real-Time-Stock-Price-Predictions-Using-LSTM.git
   cd Real-Time-Stock-Price-Predictions-Using-LSTM
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Launch the application:
   ```bash
   streamlit run app.py
   ```

## Technical Overview

### LSTM Model

- Architecture: Two LSTM layers, dropout layers, and dense output layers
- Features used: Close, SMA_20, EMA_20, RSI_14, Volatility_20, Lag1, Lag2, Volume
- Training split: 80% training / 20% testing, with validation and early stopping
- Frameworks: TensorFlow, scikit-learn

### Prediction Pipeline

1. Fetch historical stock data from 2015 to present
2. Compute technical indicators
3. Normalize and sequence the data
4. Train the model and evaluate accuracy
5. Generate forward predictions for user-specified dates

## Dependencies

- streamlit
- yfinance
- pandas
- numpy
- tensorflow
- scikit-learn
- plotly

## Acknowledgements

This application uses the Yahoo Finance API (via the `yfinance` library), Streamlit for the frontend, and Plotly for data visualization.

## License

This project is released under the MIT License. See the `LICENSE` file for details.
