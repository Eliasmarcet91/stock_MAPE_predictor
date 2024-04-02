import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

# Your Alpha Vantage API key
api_key = ''  # Add your API key here

# Function to retrieve stock data
def get_stock_data(symbol, interval='daily'):
    function = 'TIME_SERIES_DAILY' if interval == 'daily' else 'TIME_SERIES_MONTHLY'
    url = f'https://www.alphavantage.co/query?function={function}&symbol={symbol}&apikey={api_key}'
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        prices_data = data.get('Time Series (Daily)' if interval == 'daily' else 'Monthly Time Series')
        if prices_data:
            hist_data = pd.DataFrame(prices_data).T
            hist_data.index = pd.to_datetime(hist_data.index)
            hist_data.columns = [col.split()[-1] for col in hist_data.columns]  # Clean column names
            return hist_data
    return None

# Function to generate buy/sell signal based on price data
def generate_signal(row):
    if row['close'] > row['open']:
        return 'Buy'
    else:
        return 'Sell'

# Function to train a neural network model
def train_neural_network(X_train, y_train):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(1)  # Output layer (1 neuron for regression)
    ])
    model.compile(optimizer='adam', loss='mse')  # Using 'mse' instead of 'mean_squared_error'
    model.fit(X_train, y_train, epochs=100, verbose=0)
    return model

# Function to predict stock prices using the trained neural network
def predict_stock_prices(model, X_test):
    return model.predict(X_test)

# Function to analyze the trend and provide recommendation
def analyze_trend(predicted_prices):
    price_difference = predicted_prices[-1] - predicted_prices[0]
    if price_difference > 0:
        return 'Buy', predicted_prices[-1]  # If the price increased, recommend to buy
    else:
        return 'Don\'t Buy', predicted_prices[-1]

# Streamlit app
st.title('Stock Price Prediction App')

# Input stock symbol from the user
symbol = st.text_input("Enter the stock symbol (e.g., AAPL):")

if symbol:
    daily_stock_data = get_stock_data(symbol, 'daily')
    if daily_stock_data is not None:
        st.subheader('Daily Candlestick Chart')
        fig = go.Figure(data=[go.Candlestick(x=daily_stock_data.index,
                                             open=pd.to_numeric(daily_stock_data['open']),
                                             high=pd.to_numeric(daily_stock_data['high']),
                                             low=pd.to_numeric(daily_stock_data['low']),
                                             close=pd.to_numeric(daily_stock_data['close']))])
        st.plotly_chart(fig)

        # Prepare data for training the neural network
        X = daily_stock_data[['open', 'high', 'low', 'close']].shift(1).dropna()
        y = daily_stock_data['close'].shift(-1).dropna()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        # Normalize the data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train the neural network model
        model = train_neural_network(X_train_scaled, y_train)

        # Predict stock prices using the trained model
        predicted_prices = predict_stock_prices(model, X_test_scaled)

        # Analyze the trend and provide recommendation
        recommendation, predicted_price_next_day = analyze_trend(predicted_prices)
        st.write("Recommendation based on trend analysis:", recommendation)
        st.write("Predicted price for the next day:", predicted_price_next_day)

        # Measure the level of accuracy of the prediction
        mape = mean_absolute_error(y_test, predicted_prices) / np.mean(y_test)
        st.write("Mean Absolute Percentage Error (MAPE): {:.2f}%".format(mape * 100))

    else:
        st.write("Failed to retrieve stock data")
