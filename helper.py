
import ccxt
import sd  # Ensure 'sd' contains 'api_key', 'api_secret_key', 'password'
import schedule
import pandas as pd
import numpy as np
from datetime import datetime
import time
import sys
import warnings
import yfinance as yf
import io
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import mse  # Import the mse loss function

# Load the model, specifying 'mse' as a custom object if needed
try:
    restored = load_model(r'C:\Users\knath\OneDrive\Desktop\New_folder\flask\bitcoin_pred\project\my_btc_10_days_predictl.h5', custom_objects={'mse': mse})
    restored.summary()
except Exception as e:
    print(f"Error loading the model: {e}")
    sys.exit()

# Suppress warnings (use with caution)
warnings.filterwarnings('ignore')

# Initialize the exchange with real API credentials
try:
    exchange = ccxt.bitget({
        'apiKey': sd.api_key,
        'secret': sd.api_secret_key,
        'password': sd.password,
        'timeout': 30000,  # 30 seconds timeout
        'retry': 3,  # retry up to 3 times
    })
except AttributeError as e:
    print(f"Error accessing API credentials from 'sd' module: {e}. Ensure 'sd.py' exists and contains api_key, api_secret_key, and password.")
    sys.exit()
except ccxt.AuthenticationError as e:
    print(f"Authentication error with Bitget: {e}. Please check your API keys and password.")
    sys.exit()
except ccxt.NetworkError as e:
    print(f"Network error connecting to Bitget: {e}. Please check your internet connection.")
    sys.exit()
except ccxt.ExchangeError as e:
    print(f"Bitget exchange error: {e}")
    sys.exit()

symbol = "BTC/USDT"

def fetch_price():
    try:
        bars = exchange.fetch_ohlcv(symbol, timeframe='1m', limit=1)
        df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        price = df['close'].iloc[0]
        return price
    except ccxt.NetworkError as e:
        print(f"Network error fetching price: {e}")
        return None
    except ccxt.ExchangeError as e:
        print(f"Bitget exchange error fetching price: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while fetching the price: {e}")
        return None


"""
# Schedule the bot to run every 10 seconds
schedule.every(10).seconds.do(fetch_price)

while True:
    schedule.run_pending()
    time.sleep(1)
"""

def make_seq(data, seq_len):
    """
    Creates a sequence of data for prediction.

    Args:
        data (np.ndarray): The input data array.
        seq_len (int): The length of the sequence.

    Returns:
        np.ndarray: The sequence of data.
    """
    if len(data) < seq_len + 1:
        return None  # Handle cases where data is shorter than required
    x = data[:seq_len]  # Corrected slicing to get a sequence of length seq_len
    return np.array(x).reshape(1, seq_len, 1) # Reshape here for direct prediction input


def create_img(pred,num_of_days):
    """
    Creates a plot of the predicted values.

    Args:
        pred (list): The predicted values.

    Returns:
        bytes: The image in bytes format.
    """

    plt.figure(figsize=(10, 5))
    plt.plot(pred, label=f'Predicted Values ', color='red')
    plt.title(f'Predicted Values for {num_of_days} Days')
    plt.xlabel('Days')
    plt.ylabel('Price')
    plt.legend()

    # Save the plot to a BytesIO object
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)

    return img.getvalue()


def days_predict(data_input_period, num_of_days):
    """
    Predicts future values based on historical data.

    Args:
        data_input_period (int): The number of days of historical data to use.
        num_of_days (int): The number of days to predict.

    Returns:
        bytes: The image in bytes format.
    """
    n_steps = 10
    ticker = 'BTC-USD'
    try:
        recent = yf.download(ticker, period=f"{data_input_period}d", interval="1d")
        if recent.empty:
            print(f"No data downloaded for the past {data_input_period} days.")
            return None
        closes = recent['Close'].values.astype(float) # Ensure float type
        pred = []
        historical_data = closes[-n_steps:].copy() # Use the last n_steps for initial prediction

        for _ in range(num_of_days):
            if len(historical_data) < n_steps:
                # Handle cases with insufficient historical data
                padding = np.zeros(n_steps - len(historical_data))
                input_sequence = np.concatenate([padding, historical_data])
            else:
                input_sequence = historical_data[-n_steps:]

            input_sequence = input_sequence.reshape((1, n_steps, 1))
            predicted = restored.predict(input_sequence, verbose=0)
            pred.append(predicted.item())
            historical_data = np.append(historical_data, predicted.item())

        return create_img(pred,num_of_days)

    except Exception as e:
        print(f"Error during prediction: {e}")
        return None
    


