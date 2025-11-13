import tkinter as tk
from tkinter import messagebox
import pandas as pd
import numpy as np
import onnxruntime as ort

# Load the trained ONNX model
session = ort.InferenceSession('model/rf_model.onnx')

# Load the dataset
data = pd.read_csv('bitcoin_2014_to_2023.csv')
data['time'] = pd.to_datetime(data['time'])

# Function to add features (same as in the notebook)
def add_features(df):
    df = df.copy()
    df["Price_MA7"] = df["PriceUSD"].rolling(window=7).mean()
    df["Price_MA30"] = df["PriceUSD"].rolling(window=30).mean()
    df["Log_Returns"] = np.log(df["PriceUSD"] / df["PriceUSD"].shift(1))
    df["Vol_Ratio"] = df["TxCnt"] / df["AdrActCnt"]
    df = df.dropna()
    return df

# Process the data with features
data = add_features(data)

def predict_price():
    date_str = date_entry.get()
    try:
        selected_date = pd.to_datetime(date_str)
        # Find the row matching the date
        row = data[data['time'].dt.date == selected_date.date()]
        if not row.empty:
            # Get features for prediction
            features = row[['Price_MA7', 'Price_MA30', 'Log_Returns', 'Vol_Ratio']].values.astype(np.float32)
            prediction = session.run(None, {'float_input': features})[0][0]
            actual = row['PriceUSD'].values[0]
            messagebox.showinfo("Prediction Result",
                                f"Date: {selected_date.date()}\n"
                                f"Predicted Price: ${prediction:.2f}\n"
                                f"Actual Price: ${actual:.2f}")
        else:
            messagebox.showerror("Error", "Date not found in the dataset. Please enter a date between 2014-01-30 and 2023.")
    except ValueError:
        messagebox.showerror("Error", "Invalid date format. Please use YYYY-MM-DD.")

# Create the GUI
root = tk.Tk()
root.title("Bitcoin Price Predictor")

tk.Label(root, text="Enter Date (YYYY-MM-DD):").pack(pady=10)
date_entry = tk.Entry(root, width=20)
date_entry.pack(pady=5)

tk.Button(root, text="Predict", command=predict_price).pack(pady=20)

root.mainloop()
