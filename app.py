import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# Function to get stock data
def get_stock_data(ticker):
    stock_data = yf.download(ticker, start="2015-01-01", end="2024-01-01")
    stock_data['Returns'] = stock_data['Close'].pct_change()  # Calculate daily returns
    stock_data.dropna(inplace=True)
    return stock_data

# Feature Engineering
def create_dataset(data, time_step=1):
    X, y_open, y_close = [], [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), :])
        y_open.append(data[i + time_step, 0])  # Open price next day
        y_close.append(data[i + time_step, 1])  # Close price next day
    return np.array(X), np.array(y_open), np.array(y_close)

# Build LSTM Model
def build_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=2))  # Prediction for both open and close prices
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Streamlit UI
st.title("Custom Stock Price Prediction Dashboard")

# User input for stock ticker
ticker = st.text_input("Enter Stock Ticker Symbol:", "AAPL")
time_step = st.slider("Select Time Step for Prediction:", 1, 60, 30)

if st.button("Get Prediction"):
    stock_data = get_stock_data(ticker)

    # Check if data is empty
    if stock_data.empty:
        st.error("No data found for the given ticker. Please try a different ticker.")
    else:
        # Use 'Open' and 'Close' for training
        data = stock_data[['Open', 'Close']].values
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)

        # Create dataset
        X, y_open, y_close = create_dataset(scaled_data, time_step)
        X = X.reshape(X.shape[0], X.shape[1], 2)

        # Split the data
        train_size = int(len(X) * 0.7)
        X_train, X_test = X[:train_size], X[train_size:]
        y_open_train, y_open_test = y_open[:train_size], y_open[train_size:]
        y_close_train, y_close_test = y_close[:train_size], y_close[train_size:]

        # Build and train the model
        model = build_model((X_train.shape[1], 2))
        model.fit(X_train, np.column_stack((y_open_train, y_close_train)), epochs=100, batch_size=32)

        # Make predictions
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)

        # Inverse transform predictions
        train_pred_inv = scaler.inverse_transform(train_pred)
        test_pred_inv = scaler.inverse_transform(test_pred)

        # Calculate RMSE
        train_rmse_open = np.sqrt(mean_squared_error(y_open_train, train_pred_inv[:, 0]))
        train_rmse_close = np.sqrt(mean_squared_error(y_close_train, train_pred_inv[:, 1]))
        test_rmse_open = np.sqrt(mean_squared_error(y_open_test, test_pred_inv[:, 0]))
        test_rmse_close = np.sqrt(mean_squared_error(y_close_test, test_pred_inv[:, 1]))

        # Organized output section
        st.subheader("Prediction Results")
        st.write(f"### Predictions for {ticker}")
        st.write(f"**Train RMSE Open**: {train_rmse_open:.2f}, **Close**: {train_rmse_close:.2f}")
        st.write(f"**Test RMSE Open**: {test_rmse_open:.2f}, **Close**: {test_rmse_close:.2f}")

        # Show last predicted values
        st.write("### Last Predicted Values:")
        st.write(f"**Predicted Opening Price**: {test_pred_inv[-1, 0]:.2f}")
        st.write(f"**Predicted Closing Price**: {test_pred_inv[-1, 1]:.2f}")

        # Plot predictions for Open Prices
        plt.figure(figsize=(14, 7))
        plt.plot(stock_data.index[time_step:train_size + time_step], train_pred_inv[:, 0], label='Train Predicted Open', color='blue')
        plt.plot(stock_data.index[train_size + time_step:], test_pred_inv[:, 0], label='Test Predicted Open', color='orange')
        plt.plot(stock_data.index[time_step:], stock_data['Open'].values[time_step:], label='Actual Open', color='green')
        plt.title(f'Stock Open Price Prediction for {ticker}')
        plt.xlabel('Date')
        plt.ylabel('Open Price')
        plt.legend()
        st.pyplot(plt)

        # Plot predictions for Close Prices
        plt.figure(figsize=(14, 7))
        plt.plot(stock_data.index[time_step:train_size + time_step], train_pred_inv[:, 1], label='Train Predicted Close', color='blue')
        plt.plot(stock_data.index[train_size + time_step:], test_pred_inv[:, 1], label='Test Predicted Close', color='orange')
        plt.plot(stock_data.index[time_step:], stock_data['Close'].values[time_step:], label='Actual Close', color='green')
        plt.title(f'Stock Close Price Prediction for {ticker}')
        plt.xlabel('Date')
        plt.ylabel('Close Price')
        plt.legend()
        st.pyplot(plt)
