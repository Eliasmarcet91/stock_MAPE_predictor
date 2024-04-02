import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# Load the trained model
@st.cache(allow_output_mutation=True)
def load_nn_model():
    with open("https://github.com/Eliasmarcet91/stock_MAPE_predictor/raw/master/stock_prediction_MAPE.pkl", "rb") as f:
        model = pickle.load(f)
    return model

# Function to preprocess input data
def preprocess_data(data):
    # Apply the same preprocessing steps as in your training script
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    return scaled_data

# Function to make predictions
def predict_prices(model, data):
    scaled_data = preprocess_data(data)
    predictions = model.predict(scaled_data)
    return predictions

def main():
    st.title("Stock Price Prediction App")

    # Input stock symbol from the user
    symbol = st.text_input("Enter the stock symbol (e.g., AAPL):")

    # Load the model
    model = load_nn_model()

    if st.button("Predict"):
        # Get the input data (replace this with your actual input data)
        input_data = np.random.rand(1, num_features)  # Example random data
        prediction = predict_prices(model, input_data)
        st.write("Predicted Price:", prediction)

if __name__ == "__main__":
    main()
