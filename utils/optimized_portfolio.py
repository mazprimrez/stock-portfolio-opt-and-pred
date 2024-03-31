import yfinance as yf
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from cvxopt import matrix, solvers

def retrieve_stock_data(stock:str=''):
    yfin = yf.Ticker(stock)
    return yfin.history(period='5y')

def retrieve_portfolio_data(stocks:list=[]):
    portfolio_data = pd.DataFrame()
    for stock in stocks:
        data = retrieve_stock_data(stock + ".JK")
        data['Code'] = stock
        portfolio_data = pd.concat([portfolio_data, data])
    return portfolio_data[['Code', 'Close']]

def maximize_sharpe_ratio(returns_df, asset_prices, budget):
    # Calculate mean returns and covariance matrix
    asset_prices = asset_prices['Price'].to_list()
    mean_returns = returns_df.mean()
    cov_matrix = returns_df.cov()

    num_assets = len(returns_df.columns)

    # Objective function: Negative Sharpe ratio (to maximize)
    def negative_sharpe_ratio(weights):
        portfolio_return = np.dot(mean_returns, weights)
        portfolio_risk = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
        sharpe_ratio = portfolio_return / portfolio_risk
        return -sharpe_ratio

    # Constraint: Sum of values of lots <= budget
    budget_constraint = {'type': 'ineq', 'fun': lambda weights: budget - np.dot(weights, asset_prices)}
    weight_sum_constraint = {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}

    bounds = tuple((0, budget / price) for price in asset_prices)

    initial_weights = np.random.rand(num_assets) / num_assets

    # Perform optimization to maximize Sharpe ratio
    result = minimize(negative_sharpe_ratio, initial_weights, method='SLSQP', bounds=bounds, constraints=[weight_sum_constraint, budget_constraint])
    optimized_weights = result.x
    lots = np.round(optimized_weights * budget / asset_prices)

    return lots.astype(int), optimized_weights

def calculate_portfolio_return(weights, mean_returns):
    return np.dot(weights, mean_returns)

def calculate_portfolio_risk(weights, cov_matrix):
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

def calculate_annualized_return(daily_return, historical_period, trading_days_in_year=252):
    return (1 + daily_return)**(trading_days_in_year / historical_period) - 1

def preparation(data:pd.DataFrame=None):
    stock_data_raw = data.pivot_table(index="Date", columns="Code", values="Close")
    eligible_stocks = stock_data_raw.tail(7).dropna(axis=1).columns
    stock_data_raw = stock_data_raw[eligible_stocks]
    stock_data = stock_data_raw.copy()
    for col in stock_data_raw.columns:
        stock_data[col] = stock_data[col].pct_change()
    return stock_data, stock_data_raw

def get_stock_price(stock_data):
    price = stock_data.tail(7).max().T.reset_index()
    price.columns = ['Code', 'Price']
    price['Price'] = price['Price']*100
    return price

def get_stock_return(stock_data):
    return stock_data.mean()

def get_stock_risk(stock_data):
    cov_matrix = stock_data.cov()
    std_deviation = np.sqrt(np.diag(cov_matrix))
    return std_deviation

def generate_result(stock_data, optimized_weights, lots, prices, return_data, risk_data):
    result = pd.DataFrame([stock_data.columns, optimized_weights, lots], index=['Code', 'Weight', 'Lots']).T
    result = result.merge(prices, on='Code', how='left')
    result['Price'] = result['Price']/100
    result['Mean_return'] = return_data.to_list()
    result['Mean_risk'] = risk_data
    result = result.sort_values(['Lots', 'Weight'], ascending=False)
    return result.reset_index(drop=True)