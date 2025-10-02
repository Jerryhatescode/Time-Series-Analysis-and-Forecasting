
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import sqrt
import os


dataset_path = None  
date_col = "date"    
value_col = "value"  
freq = "MS"          
test_periods = 24    

def load_series(path=None):
    if path and os.path.exists(path):
        df = pd.read_csv(path)
     
        if date_col not in df.columns or value_col not in df.columns:
            raise ValueError(f"CSV must contain columns named '{date_col}' and '{value_col}'")
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col).sort_index()
        ts = df[value_col].asfreq(freq)
        return ts
    else:
        
        try:
            import statsmodels.api as sm
            ds = sm.datasets.get_rdataset("AirPassengers")
            df = ds.data
            df.columns = ['value']
            idx = pd.date_range(start='1949-01', periods=len(df), freq=freq)
            ts = pd.Series(df['value'].values, index=idx).asfreq(freq)
            return ts
        except Exception:
            # Synthesize a plausible seasonal series
            rng = pd.date_range(start='2010-01', periods=120, freq=freq)
            trend = np.linspace(50, 150, len(rng))
            seasonal = 10 * np.sin(2 * np.pi * (rng.month / 12.0))
            noise = np.random.normal(0, 5, len(rng))
            vals = trend + seasonal + noise
            ts = pd.Series(vals, index=rng).round(2)
            return ts

def visualize_series(ts):
    plt.figure(figsize=(10,4))
    plt.plot(ts)
    plt.title('Time Series - Monthly Totals')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.tight_layout()
    plt.show()

def resample_and_smooth(ts):
    annual = ts.resample('A').sum()
    rolling_12 = ts.rolling(window=12, center=False).mean()
    plt.figure(figsize=(10,4))
    plt.plot(ts, label='Monthly')
    plt.plot(rolling_12, label='12-month rolling mean')
    plt.title('Monthly Series with 12-month Rolling Mean')
    plt.legend()
    plt.tight_layout()
    plt.show()
    return annual, rolling_12

def decompose_series(ts):
    
    result = seasonal_decompose(ts, model='multiplicative', period=12, extrapolate_trend='freq')
    result.plot()
    plt.tight_layout()
    plt.show()
    return result

def fit_and_forecast(train, test, order=(1,1,1), seasonal_order=(1,1,1,12)):
    model = SARIMAX(train, order=order, seasonal_order=seasonal_order,
                    enforce_stationarity=False, enforce_invertibility=False)
    res = model.fit(disp=False)
    n_steps = len(test)
    pred = res.get_forecast(steps=n_steps)
    pred_mean = pred.predicted_mean
    pred_ci = pred.conf_int()
    return res, pred_mean, pred_ci

def evaluate_and_plot(train, test, pred_mean, pred_ci):
    mae = mean_absolute_error(test, pred_mean)
    rmse = sqrt(mean_squared_error(test, pred_mean))
    plt.figure(figsize=(10,4))
    plt.plot(train.index, train, label='Train')
    plt.plot(test.index, test, label='Test (Actual)')
    plt.plot(pred_mean.index, pred_mean, label='Forecast')
    plt.fill_between(pred_ci.index, pred_ci.iloc[:,0], pred_ci.iloc[:,1], alpha=0.2)
    plt.title(f'Forecast vs Actual (MAE={mae:.2f}, RMSE={rmse:.2f})')
    plt.legend()
    plt.tight_layout()
    plt.show()
    return mae, rmse

def main():
    ts = load_series(dataset_path)
    print("Series timeframe:", ts.index.min(), "to", ts.index.max())
    print(ts.head())

    visualize_series(ts)
    annual, rolling_12 = resample_and_smooth(ts)
    decomp = decompose_series(ts)

   
    train = ts[:-test_periods]
    test = ts[-test_periods:]

    
    res, pred_mean, pred_ci = fit_and_forecast(train, test)

    mae, rmse = evaluate_and_plot(train, test, pred_mean, pred_ci)
    print("Model summary (abbreviated):")
    print(res.summary().tables[0])
    print(f"MAE: {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")

   
    results = pd.DataFrame({'actual': test.values, 'forecast': pred_mean.values}, index=test.index)
    print("\nForecast vs Actual (first 10 rows):")
    print(results.head(10))

if __name__ == "__main__":
    main()
