#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os
from itertools import combinations
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

sigma_mult = 2

csv_files = ['/Users/dogayildiz/Desktop/20180101_20231121_bist30/20201228_20210328_bist30.csv',
              '/Users/dogayildiz/Desktop/20180101_20231121_bist30/20210329_20210627_bist30.csv',
              '/Users/dogayildiz/Desktop/20180101_20231121_bist30/20210628_20210926_bist30.csv',
              '/Users/dogayildiz/Desktop/20180101_20231121_bist30/20210927_20211226_bist30.csv',
              '/Users/dogayildiz/Desktop/20180101_20231121_bist30/20211227_20220327_bist30.csv',
              '/Users/dogayildiz/Desktop/20180101_20231121_bist30/20220328_20220626_bist30.csv',
              '/Users/dogayildiz/Desktop/20180101_20231121_bist30/20220627_20220925_bist30.csv',
              '/Users/dogayildiz/Desktop/20180101_20231121_bist30/20220926_20221225_bist30.csv',
              '/Users/dogayildiz/Desktop/20180101_20231121_bist30/20221226_20230326_bist30.csv']

# Combine CSV files and make linear interpolation for empty cells
def combine_csv(files):
    all_data = pd.DataFrame()
    for file in files:
        df = pd.read_csv(file, parse_dates=['timestamp'], index_col='timestamp')
        all_data = pd.concat([all_data, df])
    all_data.sort_index(inplace=True)  # Ensure the data is sorted by time
    all_data.interpolate(method='linear', inplace=True)  # Interpolate missing values
    return all_data

stock_data = combine_csv(csv_files)


# In[2]:


# Correlations
def calculate_correlations(df):
    pivot_df = df.pivot(columns='short_name', values='price').dropna()
    return pivot_df.corr()

correlation_matrix = calculate_correlations(stock_data)

def top_correlated_pairs(corr_matrix, num_pairs=10):
    corr_pairs = corr_matrix.unstack().sort_values(kind="quicksort", ascending=False)
    return corr_pairs[corr_pairs != 1].drop_duplicates().head(num_pairs)

print("Top 10 Correlated Pairs:")
print(top_correlated_pairs(correlation_matrix))


# In[3]:


stock1 = 'KCHOL'
stock2 = 'SAHOL'

def linear_regression_model(df, stock1, stock2):
    X = df[df['short_name'] == stock1]['price'].values.reshape(-1, 1)
    y = df[df['short_name'] == stock2]['price'].values
    model = LinearRegression().fit(X, y)
    return model

model = linear_regression_model(stock_data, stock1, stock2)

def calculate_residuals(df, model, start_date, end_date):
    filtered_df = df[(df.index >= start_date) & (df.index <= end_date)]
    X = filtered_df[filtered_df['short_name'] == stock1]['price'].values.reshape(-1, 1)
    y_actual = filtered_df[filtered_df['short_name'] == stock2]['price'].values
    y_predicted = model.predict(X)
    residuals = y_actual - y_predicted
    return residuals, filtered_df

residuals, filtered_data = calculate_residuals(stock_data, model, '2023-01-01', '2023-03-31')


# In[4]:


# Plotting residuals with control limits
def plot_residuals_with_control_limits(residuals, filtered_data, ucl ,lcl):
    plt.figure(figsize=(10, 6))
    plt.plot(filtered_data[filtered_data['short_name'] == stock2].index, residuals, label='Residuals')
    plt.axhline(y=upper_control_limit, color='r', linestyle='--', label='Upper Control Limit')
    plt.axhline(y=mean_residual, color='g', linestyle='-', label='Mean')
    plt.axhline(y=lower_control_limit, color='b', linestyle='--', label='Lower Control Limit')
    plt.title(f'Residuals Plot for {stock1} and {stock2}')
    plt.xlabel('Date')
    plt.ylabel('Residual')
    plt.legend()
    plt.show()

mean_residual = np.mean(residuals)
std_residual = np.std(residuals)
upper_control_limit = mean_residual + sigma_mult * std_residual
lower_control_limit = mean_residual - sigma_mult * std_residual

plot_residuals_with_control_limits(residuals, filtered_data, upper_control_limit, lower_control_limit)


# In[5]:


# BUY or SELL signals
def generate_trading_signals(filtered_data, residuals):
    signals = []
    stock1_data = filtered_data[filtered_data['short_name'] == stock1]
    stock2_data = filtered_data[filtered_data['short_name'] == stock2]

    for index, residual in enumerate(residuals):
        timestamp = stock1_data.index[index]
        stock1_price = stock1_data['price'].iloc[index]
        stock2_price = stock2_data['price'].iloc[index]

        if residual > upper_control_limit:
            signals.append((timestamp, stock1_price, 'BUY', stock2_price, 'SELL'))
        elif residual < lower_control_limit:
            signals.append((timestamp, stock1_price, 'SELL', stock2_price, 'BUY'))
    return signals

trading_signals = generate_trading_signals(filtered_data, residuals)

print(f"{'Date':<12} {'Hour':<8} {'KCHOL Price':<12} {'KCHOL Decision':<15} {'SAHOL Price':<12} {'SAHOL Decision'}")
for signal in trading_signals:
    date_str = signal[0].date().strftime('%Y-%m-%d')
    time_str = signal[0].time().strftime('%H:%M:%S')
    print(f"{date_str:<12} {time_str:<8} {signal[1]:<12.2f} {signal[2]:<15} {signal[3]:<12.2f} {signal[4]}")


# In[6]:


# Function to simulate the trading based on signals
def simulate_trading(signals, initial_capital):
    capital = initial_capital
    sahol_shares = 0
    kchol_shares = 0
    current_stock = None  # Track which stock is currently being held

    for signal in signals:
        date, kchol_price, kchol_decision, sahol_price, sahol_decision = signal

        if current_stock is None:
            # First signal: decide which stock to buy initially
            if sahol_decision == 'BUY':
                sahol_shares = capital // sahol_price
                capital -= sahol_shares * sahol_price
                current_stock = 'SAHOL'
            elif kchol_decision == 'BUY':
                kchol_shares = capital // kchol_price
                capital -= kchol_shares * kchol_price
                current_stock = 'KCHOL'

        else:
            # Subsequent signals: execute trades based on the current stock
            if current_stock == 'SAHOL' and kchol_decision == 'BUY':
                # Sell SAHOL and buy KCHOL
                capital += sahol_shares * sahol_price
                sahol_shares = 0
                kchol_shares = capital // kchol_price
                capital -= kchol_shares * kchol_price
                current_stock = 'KCHOL'

            elif current_stock == 'KCHOL' and sahol_decision == 'BUY':
                # Sell KCHOL and buy SAHOL
                capital += kchol_shares * kchol_price
                kchol_shares = 0
                sahol_shares = capital // sahol_price
                capital -= sahol_shares * sahol_price
                current_stock = 'SAHOL'

    # Calculate final value of the portfolio
    final_portfolio_value = capital + sahol_shares * sahol_price + kchol_shares * kchol_price
    profit_or_loss = final_portfolio_value - initial_capital
    return profit_or_loss

# Use the trading signals generated in your existing code
profit_or_loss = simulate_trading(trading_signals, 100000)

print(f"Profit or Loss at the end of the simulation: {profit_or_loss}₺")


# In[7]:


stock1 = 'AKBNK'
stock2 = 'YKBNK'
model = linear_regression_model(stock_data, stock1, stock2)
residuals, filtered_data = calculate_residuals(stock_data, model, '2023-01-01', '2023-03-31')
mean_residual = np.mean(residuals)
std_residual = np.std(residuals)
upper_control_limit = mean_residual + sigma_mult * std_residual
lower_control_limit = mean_residual - sigma_mult * std_residual

plot_residuals_with_control_limits(residuals, filtered_data, upper_control_limit, lower_control_limit)


# In[8]:


trading_signals = generate_trading_signals(filtered_data, residuals)

print(f"{'Date':<12} {'Hour':<8} {'AKBNK Price':<12} {'AKBNK Decision':<15} {'YKBNK Price':<12} {'YKBNK Decision'}")
for signal in trading_signals:
    date_str = signal[0].date().strftime('%Y-%m-%d')
    time_str = signal[0].time().strftime('%H:%M:%S')
    print(f"{date_str:<12} {time_str:<8} {signal[1]:<12.2f} {signal[2]:<15} {signal[3]:<12.2f} {signal[4]}")


# In[9]:


# Use the trading signals generated in your existing code
profit_or_loss = simulate_trading(trading_signals, 100000)

print(f"Profit or Loss at the end of the simulation: {profit_or_loss}₺")


# In[10]:


import statsmodels.api as sm

stock1 = 'KCHOL'
stock2 = 'SAHOL'

# Perform time series analysis 
def time_series_analysis(df, stock1, stock2):
    X = df[df['short_name'] == stock1]['price'].values.reshape(-1, 1)
    y = df[df['short_name'] == stock2]['price'].values

    model = LinearRegression().fit(X, y)

    residuals = y - model.predict(X)

    # Time series modeling (ARIMA)
    ts_model = sm.tsa.ARIMA(residuals, order=(1, 0, 1)).fit()

    forecasted_residuals = ts_model.predict(start=len(residuals), end=len(residuals) + len(X) - 1)

    mean_residual = np.mean(residuals)
    std_residual = np.std(residuals)
    updated_control_limits = (mean_residual + sigma_mult * std_residual, mean_residual - sigma_mult * std_residual)

    return model, updated_control_limits

new_model, new_control_limits = time_series_analysis(stock_data, stock1, stock2)


# In[11]:


def calculate_residuals(df, model, start_date, end_date):
    filtered_df = df[(df.index >= start_date) & (df.index <= end_date)]
    X = filtered_df[filtered_df['short_name'] == stock1]['price'].values.reshape(-1, 1)
    y_actual = filtered_df[filtered_df['short_name'] == stock2]['price'].values
    y_predicted = model.predict(X)
    residuals = y_actual - y_predicted
    return residuals, filtered_df

new_residuals, new_filtered_data = calculate_residuals(stock_data, new_model, '2023-01-01', '2023-03-31')


# In[12]:


# Plotting residuals with control limits
def plot_residuals_with_control_limits(residuals, filtered_data, control_limits):
    plt.figure(figsize=(10, 6))
    plt.plot(filtered_data[filtered_data['short_name'] == stock2].index, residuals, label='Residuals')
    plt.axhline(y=control_limits[0], color='r', linestyle='--', label='Upper Control Limit')
    plt.axhline(y=0, color='g', linestyle='-', label='Mean')
    plt.axhline(y=control_limits[1], color='b', linestyle='--', label='Lower Control Limit')
    plt.title(f'Residuals Plot for {stock1} and {stock2}')
    plt.xlabel('Date')
    plt.ylabel('Residual')
    plt.legend()
    plt.show()

plot_residuals_with_control_limits(new_residuals, new_filtered_data, new_control_limits)


# In[13]:


def generate_trading_signals(filtered_data, residuals, control_limits):
    signals = []
    stock1_data = filtered_data[filtered_data['short_name'] == stock1]
    stock2_data = filtered_data[filtered_data['short_name'] == stock2]

    for index, residual in enumerate(residuals):
        timestamp = stock1_data.index[index]
        stock1_price = stock1_data['price'].iloc[index]
        stock2_price = stock2_data['price'].iloc[index]

        if residual > control_limits[0]:
            signals.append((timestamp, stock1_price, 'BUY', stock2_price, 'SELL'))
        elif residual < control_limits[1]:
            signals.append((timestamp, stock1_price, 'SELL', stock2_price, 'BUY'))
    return signals

# Display signals
new_trading_signals = generate_trading_signals(new_filtered_data, new_residuals, new_control_limits)
print(f"{'Date':<12} {'Hour':<8} {'KCHOL Price':<12} {'KCHOL Decision':<15} {'SAHOL Price':<12} {'SAHOL Decision'}")
for signal in new_trading_signals:
    date_str = signal[0].date().strftime('%Y-%m-%d')
    time_str = signal[0].time().strftime('%H:%M:%S')
    print(f"{date_str:<12} {time_str:<8} {signal[1]:<12.2f} {signal[2]:<15} {signal[3]:<12.2f} {signal[4]}")


# In[14]:


def simulate_trading(signals, initial_capital):
    capital = initial_capital
    sahol_shares = 0
    kchol_shares = 0
    current_stock = None

    for signal in signals:
        date, kchol_price, kchol_decision, sahol_price, sahol_decision = signal

        if current_stock is None:
            if sahol_decision == 'BUY':
                sahol_shares = capital // sahol_price
                capital -= sahol_shares * sahol_price
                current_stock = 'SAHOL'
            elif kchol_decision == 'BUY':
                kchol_shares = capital // kchol_price
                capital -= kchol_shares * kchol_price
                current_stock = 'KCHOL'
        else:
            if current_stock == 'SAHOL' and kchol_decision == 'BUY':
                capital += sahol_shares * sahol_price
                sahol_shares = 0
                kchol_shares = capital // kchol_price
                capital -= kchol_shares * kchol_price
                current_stock = 'KCHOL'
            elif current_stock == 'KCHOL' and sahol_decision == 'BUY':
                capital += kchol_shares * kchol_price
                kchol_shares = 0
                sahol_shares = capital // sahol_price
                capital -= sahol_shares * sahol_price
                current_stock = 'SAHOL'

    final_portfolio_value = capital + sahol_shares * sahol_price + kchol_shares * kchol_price
    profit_or_loss = final_portfolio_value - initial_capital
    return profit_or_loss

initial_capital = 100000  # 100,000₺ initial capital
profit_or_loss = simulate_trading(new_trading_signals, initial_capital)

print(f"Profit or Loss at the end of the simulation: {profit_or_loss}₺")


# In[15]:


stock1 = 'AKBNK'
stock2 = 'YKBNK'

new_model, new_control_limits = time_series_analysis(stock_data, stock1, stock2)
new_residuals, new_filtered_data = calculate_residuals(stock_data, new_model, '2023-01-01', '2023-03-31')
plot_residuals_with_control_limits(new_residuals, new_filtered_data, new_control_limits)


# In[16]:


# Display signals
new_trading_signals = generate_trading_signals(new_filtered_data, new_residuals, new_control_limits)
print(f"{'Date':<12} {'Hour':<8} {'AKBNK Price':<12} {'AKBNK Decision':<15} {'YKBNK Price':<12} {'YKBNK Decision'}")
for signal in new_trading_signals:
    date_str = signal[0].date().strftime('%Y-%m-%d')
    time_str = signal[0].time().strftime('%H:%M:%S')
    print(f"{date_str:<12} {time_str:<8} {signal[1]:<12.2f} {signal[2]:<15} {signal[3]:<12.2f} {signal[4]}")


# In[17]:


initial_capital = 100000  # 100,000₺ initial capital
profit_or_loss = simulate_trading(new_trading_signals, initial_capital)

print(f"Profit or Loss at the end of the simulation: {profit_or_loss}₺")

