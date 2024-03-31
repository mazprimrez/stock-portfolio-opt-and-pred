
import streamlit as st
import numpy as np
from sklearn.metrics import root_mean_squared_error, mean_absolute_percentage_error

def scoring(forecast, hist, return_proba, stock_ticker):
    forecast_tmp = forecast.copy()
    forecast_tmp['ds'] = forecast_tmp['ds'].astype(str)

    hist_tmp = hist.copy()
    hist_tmp['ds'] = hist_tmp['ds'].astype(str)

    result = hist_tmp.merge(forecast_tmp[['ds','yhat']], on='ds', how='right')
    result_hist = result.dropna()
    result_future = result[result['y'].isna()]

    for i in [7, 30, 90, 365]:
        return_proba['stock'].append(stock_ticker)
        return_proba['period'].append(i)

        present = result_future.head(i).tail(1)['yhat'].values[0]
        past = result_hist.tail(1)['y'].values[0]

        tmp = result_hist.tail(i)
        rmse = np.round(root_mean_squared_error(tmp['y'], tmp['yhat']), 2)
        mape = np.round(mean_absolute_percentage_error(tmp['y'], tmp['yhat']), 2)

        return_proba[f'return'].append((present - past)/past * 100)
        return_proba[f'rmse'].append(rmse)
        return_proba[f'mape'].append(mape)
    return return_proba, result_hist, result_future

def chart_preparation(result_hist, result_future, stock_ticker):
    st.line_chart(result_hist, x='ds', y='y', color="blue")
    st.line_chart(result_hist, x='ds', y='yhat', color="green")
    st.line_chart(result_future, x='ds', y='yhat', color="red")