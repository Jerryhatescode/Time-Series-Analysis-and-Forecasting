📊 Time Series Forecasting with SARIMAX
---------------------------------------

This script demonstrates an end-to-end time series forecasting pipeline
using Pandas, Statsmodels, and Scikit-learn. 

It covers:
- Loading dataset (CSV or AirPassengers or synthetic seasonal data)
- Visualization
- Resampling & smoothing
- Seasonal decomposition
- Forecasting with SARIMAX
- Evaluation with MAE & RMSE

⚙️ Requirements:
    pip install pandas numpy matplotlib statsmodels scikit-learn

📑 Usage:
    1. Run with default dataset:
        python time_series_forecast.py

    2. Run with your dataset:
        - Prepare a CSV with columns: `date`, `value`
        - Update `dataset_path = "your_dataset.csv"` in the script
        - Run:
            python time_series_forecast.py

Workflow:
    1. Load series  → load_series()
    2. Visualize    → visualize_series()
    3. Smooth/resample → resample_and_smooth()
    4. Decompose    → decompose_series()
    5. Train/Test split
    6. SARIMAX fit  → fit_and_forecast()
    7. Evaluate     → evaluate_and_plot()

Output:
    - Plots (raw series, decomposition, forecasts)
    - Model summary
    - MAE & RMSE
    - Forecast vs Actual values
