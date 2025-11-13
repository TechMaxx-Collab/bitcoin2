# Bitcoin Price Prediction (2014–2024)

---

##  Overview

This project applies **machine learning techniques** to predict **Bitcoin (BTC-USD)** prices using historical data (2014–2024).
It explores **time-series forecasting** through models like Ridge Regression, Random Forest, and XGBoost, with full performance evaluation and visualization.

Data, models, and outputs are stored persistently using **Google Drive** integration when running in **Google Colab**.

---

## 📂 Project Structure

```
bitcoin/
├── BTC-USD (2014–2024).csv              # Original dataset
├── model/
│   ├── best_model_Ridge_Tuned.pkl       # Trained Ridge Regression model
│   ├── model_metadata.json              # Feature metadata for inference
│   └── best_model_Ridge_Tuned.onnx      # Attempted ONNX export
└── output/
    ├── performance_plot_5min.png
    ├── evaluation_report_5min.csv
    ├── performance_plot_15min.png
    ├── evaluation_report_15min.csv
    ├── performance_plot_1hour.png
    ├── evaluation_report_1hour.csv
    └── final_evaluation_summary.csv
```

---

##  Objective

> **Goal:** Predict Bitcoin’s closing price using past data and engineered time-series features to identify trends and patterns.

---

##  Tech Stack

| Category              | Tools / Libraries           |
| --------------------- | --------------------------- |
| **Language**          | Python 3.10+                |
| **Data Processing**   | pandas, numpy               |
| **Machine Learning**  | scikit-learn, xgboost       |
| **Visualization**     | matplotlib, seaborn, plotly |
| **Model Persistence** | joblib, json                |
| **Environment**       | Google Colab + Google Drive |

---

## 🔬 Machine Learning Pipeline

1. **Data Loading** — Import CSV and clean missing values
2. **Feature Engineering** — Generate moving averages, volatility, log returns, and lags
3. **Scaling** — Normalize features using `StandardScaler`
4. **Model Training** — Fit Linear, RandomForest, Ridge, and XGBoost models
5. **Hyperparameter Tuning** — Optimize using `RandomizedSearchCV` and `GridSearchCV`
6. **Evaluation** — Compare models via R², RMSE, and MAPE
7. **Persistence** — Save the best model and metadata with `joblib`

---

##  Best Model

| Model                        | R²     | RMSE     | MAPE     | Description                                           |
| ---------------------------- | ------ | -------- | -------- | ----------------------------------------------------- |
| **Ridge Regression (Tuned)** | ✅ Best | ✅ Lowest | ✅ Lowest | Robust linear model after hyperparameter optimization |

---

##  Sample Output Preview

### Performance Plot (5-Minute Forecast)

### Evaluation Summary

| Metric   | Value (Example) |
| -------- | --------------- |
| **R²**   | 0.94            |
| **RMSE** | 155.67          |
| **MAPE** | 1.83%           |

*(Values are indicative — refer to **`/output/final_evaluation_summary.csv`** for full metrics.)*

---

##  Model Usage

Load the saved model in Python:

```python
from joblib import load
import json

# Load model
model = load("model/best_model_Ridge_Tuned.pkl")

# Load feature metadata
with open("model/model_metadata.json") as f:
    meta = json.load(f)
features = meta["features"]
```

Ensure that the **input feature columns** match those in `model_metadata.json`.

---

##  Key Concepts

* **Time Series Forecasting** – Sequential prediction of price movements
* **Feature Engineering** – Deriving meaningful metrics (MA, volatility, ratios)
* **Sequential Train-Test Split** – Avoids data leakage from future timestamps
* **Cross-Validation (TimeSeriesSplit)** – Realistic model evaluation
* **ONNX Conversion** – Enables framework-independent model deployment

---

##  Outputs

| File                           | Purpose                               |
| ------------------------------ | ------------------------------------- |
| `evaluation_report_5min.csv`   | Metrics for 5-min predictions         |
| `evaluation_report_15min.csv`  | Metrics for 15-min predictions        |
| `evaluation_report_1hour.csv`  | Metrics for 1-hour predictions        |
| `performance_plot_*.png`       | Predicted vs actual visualization     |
| `final_evaluation_summary.csv` | Consolidated report for all intervals |

---

##  Deployment Ideas

* Convert model to **ONNX** for interoperability
* Host API using **FastAPI** or **Flask**
* Deploy on **AWS SageMaker**, **GCP AI Platform**, or **Hugging Face Spaces**
* Integrate with a **real-time trading dashboard** or **Telegram bot**

---

##  Limitations

* Financial markets are **volatile and non-stationary**
* Ridge Regression assumes **linearity**, limiting performance on complex patterns
* Model performance may degrade over **time without retraining**
* **External events** (news, regulations, sentiment) not accounted for

---

##  Future Enhancements

* Incorporate **LSTM / Transformer models** for deeper temporal learning
* Add **technical indicators** like RSI, MACD, and Bollinger Bands
* Integrate **real-time crypto APIs** for live forecasting
* Build a **web dashboard** using Streamlit or Dash
* Enable **automated retraining pipelines** for production use

---

##  Author

**Developed by:** [TechMaxx Team&Co](https://github.com/)
 *Machine Learning & Financial Forecasting Enthusiast*
 *Project Duration:* 2014–2024 Dataset Study
 [Google Colab Notebook](https://colab.research.google.com/drive/1gcklGqHs1DgLLMWXqa7PnznRrC4JhpZk)

---

##  License

This project is licensed under the **MIT License** — you’re free to use, modify, and distribute with attribution.

---

⭐ **If you find this useful, don’t forget to star the repository!**
