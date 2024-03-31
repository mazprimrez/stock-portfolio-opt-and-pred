import streamlit as st
from prophet import Prophet
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from utils.optimized_portfolio import calculate_annualized_return, calculate_portfolio_return, calculate_portfolio_risk, generate_result, get_stock_price, get_stock_return, get_stock_risk, maximize_sharpe_ratio, preparation, retrieve_portfolio_data
from utils.price_prediction import scoring

st.set_page_config(layout="wide")
st.title("Mazi's Personal Portfolio Stock Optimization and Prediction")
st.write("Stock optimization is based on Markowitz theory and stock prediction is based on Prophet algorithm.")

input_stocks, input_budget, pred_button = st.columns(3)

with input_stocks:
    stocks = st.text_input('Input all Indonesian stock codes you plan to buy seperated by comma (e.g. BBCA, BMRI, ACES)', None)
with input_budget:
    budget = st.number_input('Insert your budget in IDR', 500000)
with pred_button:
    st.write("Click here if you want to predict the stock prices using Prophet")
    if st.button('Predict Stock Prices'):
        predict_ = True

opt, pred = st.columns(2)

with opt:
    if stocks:
        stocks = stocks.split(', ')
        portfolio_data = retrieve_portfolio_data(stocks)

        stock_data, stock_data_raw = preparation(portfolio_data)

        asset_prices = get_stock_price(stock_data_raw)
        return_data = get_stock_return(stock_data)
        risk_data = get_stock_risk(stock_data)
        
        cov_matrix = stock_data.cov()
        historical_period = stock_data.shape[0]
        lots, optimized_weights = maximize_sharpe_ratio(stock_data, asset_prices, budget)

        portfolio_return = calculate_portfolio_return(optimized_weights, return_data.mean())
        portfolio_risk = calculate_portfolio_risk(optimized_weights, cov_matrix)
        portoflio_annual_return = calculate_annualized_return(portfolio_return, historical_period)

        res = generate_result(stock_data, optimized_weights, lots, asset_prices, return_data, risk_data)

        st.write("Result of how you should allocate your money")
        st.table(res)

with pred:
    try:
        if predict_:
            return_proba = {'stock':[],
                        'period':[], 'rmse':[], 'mape':[], 'return':[]}
            hi_res = pd.DataFrame()
            fu_res = pd.DataFrame()
            

            for stock_ticker in stock_data_raw.columns:
                hist = stock_data_raw[stock_ticker].reset_index()
                hist = hist.rename({'Date': 'ds', stock_ticker: 'y'}, axis='columns')
                hist['ds'] = hist['ds'].dt.date

                m = Prophet()
                m.fit(hist)
                future = m.make_future_dataframe(periods=365)
                forecast = m.predict(future)

                return_proba, result_hist, result_future = scoring(forecast, hist, return_proba, stock_ticker)
                
                result_hist['Code'] = stock_ticker
                result_future['Code'] = stock_ticker

                hi_res = pd.concat([hi_res, result_hist])
                fu_res = pd.concat([fu_res, result_future])

            res_2 = pd.DataFrame(return_proba)
            res_2 = res_2.pivot_table(index='stock', columns='period', values=['rmse', 'mape', 'return'])
            st.table(res_2)
    except NameError:
        pass



# line chat
try:
    eligible_stocks = res[res['Lots']>0]['Code'].values
    if predict_ & len(eligible_stocks)>0:
        
        row = int(np.ceil(len(eligible_stocks)/3))
        col = 3
        if row > 1:
            fig, ax = plt.subplots(ncols=col, nrows=row, figsize=(20,15))
        else:
            fig, ax = plt.subplots(ncols=col, nrows=row, figsize=(20,5))
            
        for i, ax in enumerate(fig.axes):
            ax.set_title(f"Prediction of {eligible_stocks[i]}")
            sns.lineplot(data=hi_res[hi_res['Code']==eligible_stocks[i]], x='ds', y='y', ax=ax, color='black', label='actual')
            sns.lineplot(data=hi_res[hi_res['Code']==eligible_stocks[i]], x='ds', y='yhat', ax=ax, color='blue', label='historical_prediction')
            sns.lineplot(data=fu_res[fu_res['Code']==eligible_stocks[i]], x='ds', y='yhat', ax=ax, color='red', label='prediciton')
            ax.set_xticks([])
            if i >= len(eligible_stocks) - 1:
                break

        st.pyplot(fig)
except NameError:
    pass
