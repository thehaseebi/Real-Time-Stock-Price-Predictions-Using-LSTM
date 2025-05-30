# models.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import streamlit as st
import os


os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

def calculate_technical_indicators(df):
    """Add feature engineering: SMA, EMA, RSI, volatility, and lagged features."""
    df = df.copy()

    # Simple Moving Average (20-day)
    df['SMA_20'] = df['Close'].rolling(window=20).mean()

    # Exponential Moving Average (20-day)
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()

    # Relative Strength Index (14-day)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI_14'] = 100 - (100 / (1 + rs))

    # Volatility (20-day standard deviation of returns)
    df['Returns'] = df['Close'].pct_change()
    df['Volatility_20'] = df['Returns'].rolling(window=20).std()

    # Lagged prices (1-day and 2-day lag)
    df['Lag1'] = df['Close'].shift(1)
    df['Lag2'] = df['Close'].shift(2)

    # Include volume
    df['Volume'] = df['Volume']

    # Drop NaN values from feature engineering
    df = df.dropna()
    return df

@st.cache_data
def predict_with_lstm(df, future_date, sequence_length=60):
    """Predict the stock price for a future date using an enhanced LSTM model."""
    try:
        # Step 1: Data Preparation and Feature Engineering
        df = df.reset_index()
        df = calculate_technical_indicators(df)

        if df.empty:
            raise ValueError("No valid data after feature engineering. Ensure sufficient historical data.")

        # Features to use for prediction
        features = ['Close', 'SMA_20', 'EMA_20', 'RSI_14', 'Volatility_20', 'Lag1', 'Lag2', 'Volume']
        data = df[features].values

        # Step 2: Data Transformation - Scaling
        scaler = MinMaxScaler()
        data_scaled = scaler.fit_transform(data)

        # Step 3: Create Sequences for LSTM
        X, y = [], []
        for i in range(len(data_scaled) - sequence_length):
            X.append(data_scaled[i:i + sequence_length])
            y.append(data_scaled[i + sequence_length, 0])  # Predict 'Close' price
        X = np.array(X)
        y = np.array(y)

        if len(X) < 10:  # Ensure enough sequences for training
            raise ValueError(
                f"Insufficient data for LSTM training. Need at least {sequence_length + 10} rows, got {len(df)}.")

        # Step 4: Split Data (80% train, 20% test, no shuffle for time series)
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        # Step 5: Build LSTM Model
        model = Sequential()
        model.add(LSTM(units=100, return_sequences=True, input_shape=(sequence_length, len(features))))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50))
        model.add(Dropout(0.2))
        model.add(Dense(units=25))
        model.add(Dense(units=1))  # Output single price
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

        # Step 6: Train Model with Early Stopping
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        model.fit(X_train, y_train, epochs=100, batch_size=64, validation_split=0.1,
                  callbacks=[early_stopping], verbose=1)

        # Step 7: Evaluate Model
        y_pred = model.predict(X_test, verbose=0)
        from sklearn.metrics import mean_absolute_error
        mae = mean_absolute_error(y_test, y_pred)
        print(f"LSTM Model MAE: {mae:.4f}")

        # Step 8: Prepare Input for Future Prediction
        last_sequence = data_scaled[-sequence_length:]
        days_ahead = (future_date - df['Date'].iloc[-1].date()).days
        if days_ahead < 0:
            raise ValueError(
                f"Future date {future_date} is before the last historical date {df['Date'].iloc[-1].date()}")

        # Step 9: Iterative Prediction for Future Days
        current_sequence = last_sequence.copy()
        for _ in range(days_ahead):
            current_sequence_reshaped = current_sequence.reshape((1, sequence_length, len(features)))
            next_pred_scaled = model.predict(current_sequence_reshaped, verbose=0)

            # Create a new feature vector with predicted Close price
            # Approximate other features (use last known values or simple rules)
            new_features = np.zeros(len(features))
            new_features[0] = next_pred_scaled[0, 0]  # Predicted Close
            # Update SMA_20, EMA_20, RSI_14, Volatility_20, Lag1, Lag2, Volume with last known values
            new_features[1:] = current_sequence[-1, 1:]  # Use last features as approximation
            current_sequence = np.vstack((current_sequence[1:], new_features))

        # Step 10: Inverse Transform Prediction
        predicted_price_scaled = current_sequence[-1, 0]
        # Create a dummy array for inverse scaling (only Close matters)
        dummy_array = np.zeros((1, len(features)))
        dummy_array[0, 0] = predicted_price_scaled
        predicted_price = scaler.inverse_transform(dummy_array)[0, 0]

        return predicted_price
    except Exception as e:
        raise Exception(f"Error predicting with LSTM: {str(e)}")

