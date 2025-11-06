
# Predictive Modeling on Synthetic Bitcoin Data using Regression Techniques

This project focuses on building and evaluating regression-based machine learning models to predict synthetic Bitcoin price movements. It is designed for a broad audience, including beginners in Machine Learning, data science practitioners, crypto analysts, and recruiters evaluating ML project work.

---

##  Project Overview

The goal of this project is to explore how different regression techniques perform in predicting synthetic Bitcoin price data. The project includes data preprocessing, feature engineering, model training, evaluation, and visualization of results. In addition, the project provides instructions for retraining the model and deploying it using Streamlit.

---

##  Folder Structure

```
bitcoin2/
│
├── bitcoin_2014_to_2023.csv          # Synthetic Bitcoin dataset
├── bitcoin2.ipynb                     # Jupyter Notebook with full workflow
│
├── model/
│   ├── rf_model.onnx                  # Exported Random Forest Model in ONNX format
│   └── rf_model.pkl                   # Saved Random Forest Model (Pickle)
│
├── output/
│   ├── evaluation_report.csv          # Model performance results
│   ├── performance_plot.png           # Performance visualization
│   ├── performance_plot_5-min.png
│   ├── performance_plot_15-min.png
│   ├── performance_plot_1-hour.png
│
└── notebook/                          # Additional notebook space
```

---

##  Models Used

The project evaluates and compares the performance of the following regression models:

| Model | Description |
|--------|----------------------------|
| Linear Regression | Baseline model for comparison |
| Random Forest Regressor | Ensemble method for improved performance |
| XGBoost Regressor | Boosted tree model for advanced performance |

---

##  System Architecture

```mermaid
flowchart TD
A[Data Source: Synthetic BTC Data] --> B[Data Preprocessing & Feature Engineering]
B --> C[Train-Test Split]
C --> D[Train Regression Models]
D --> E[Model Evaluation & Metrics]
E --> F[Save Best Model (.pkl, .onnx)]
F --> G[Streamlit Deployment]
```

---

##  Model Performance

Evaluation metrics used:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

A comparison summary is available in:  
`output/evaluation_report.csv`

Performance plots provide visual insight into model accuracy and prediction trends over different time intervals.

---

##  Results Visualizations

Performance plots generated during evaluation include:

- Overall model performance comparison
- 5-minute, 15-minute, and 1-hour prediction intervals

These are located in the `output/` folder.

---

##  Installation & Setup

### **Prerequisites**

Ensure you have the following installed:

- Python 3.8+
- pip
- Virtual environment (recommended)

### **Install Dependencies**

```bash
pip install -r requirements.txt
```

---

##  Usage

### **Run Jupyter Notebook**

To explore the complete workflow:

```bash
jupyter notebook bitcoin2.ipynb
```

---

##  Retraining the Model

To retrain the model:

1. Load the dataset
2. Run preprocessing steps
3. Train the three regression models
4. Evaluate performance and export the best model

The notebook explains each step in detail.

---

##  Streamlit Deployment

To deploy the model via Streamlit:

1. Create a `app.py` file that loads the model
2. Build UI for inputting Bitcoin features
3. Display prediction results and charts

Run the app using:

```bash
streamlit run app.py
```

This allows real-time Bitcoin price prediction via a web interface.

---

##  License

This project is licensed under the **MIT License**.

---

###  Future Enhancements

- Incorporate LSTM and deep learning models
- Real-time data ingestion (e.g., Binance API)
- Add Docker deployment
- Build a web dashboard for investors

---

If you'd like, I can also generate a **requirements.txt** + **Streamlit app.py** template for deployment support.

