from flask import Flask, request, render_template, jsonify, send_file
import pandas as pd
import numpy as np
import onnxruntime as ort
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

@app.route('/', methods=['GET'])
def home():
    logger.info(f"Request received: {request.method} {request.path}")
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    logger.info(f"Request received: {request.method} {request.path}")
    date_str = request.form.get('date')
    if not date_str:
        return jsonify({'error': 'Date is required.'}), 400
    try:
        selected_date = pd.to_datetime(date_str)
        # Find the row matching the date
        row = data[data['time'].dt.date == selected_date.date()]
        if not row.empty:
            # Get features for prediction
            features = row[['Price_MA7', 'Price_MA30', 'Log_Returns', 'Vol_Ratio']].values.astype(np.float32)
            prediction = session.run(None, {'float_input': features})[0][0]
            actual = row['PriceUSD'].values[0]

            # Generate charts
            # Prediction chart
            plt.figure(figsize=(12, 6))
            plt.plot(data['time'], data['PriceUSD'], label='Actual Price', color='blue', alpha=0.7)

            # Get predictions for all data points
            features_all = data[['Price_MA7', 'Price_MA30', 'Log_Returns', 'Vol_Ratio']].values.astype(np.float32)
            predictions_all = session.run(None, {'float_input': features_all})[0].flatten()

            plt.plot(data['time'], predictions_all, label='Predicted Price', color='orange', alpha=0.7)
            plt.title('Bitcoin Price: Actual vs Predicted')
            plt.xlabel('Date')
            plt.ylabel('Price (USD)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            # Save chart to base64
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100)
            buf.seek(0)
            chart_base64 = base64.b64encode(buf.read()).decode('utf-8')
            plt.close()

            # Correlation heatmap
            plt.figure(figsize=(10, 8))
            numeric_df = data.select_dtypes(include=[np.number])
            sns.heatmap(numeric_df.corr(), cmap='coolwarm', annot=True, fmt='.2f', square=True)
            plt.title('Feature Correlation Heatmap')
            plt.tight_layout()

            # Save heatmap to base64
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100)
            buf.seek(0)
            heatmap_base64 = base64.b64encode(buf.read()).decode('utf-8')
            plt.close()

            return jsonify({
                'date': selected_date.date().isoformat(),
                'predicted_price': round(float(prediction), 2),
                'actual_price': round(float(actual), 2),
                'chart': f'data:image/png;base64,{chart_base64}',
                'heatmap': f'data:image/png;base64,{heatmap_base64}'
            })
        else:
            return jsonify({'error': 'Date not found in the dataset. Please enter a date between 2014-01-30 and 2023.'}), 404
    except ValueError:
        return jsonify({'error': 'Invalid date format. Please use YYYY-MM-DD.'}), 400



if __name__ == '__main__':
    app.run(debug=True)
