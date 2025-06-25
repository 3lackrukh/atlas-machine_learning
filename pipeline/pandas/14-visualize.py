#!/usr/bin/env python3
"""Module for visualizing DataFrame with transformations and plotting"""
import matplotlib.pyplot as plt
import pandas as pd
from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

# Remove the Weighted_Price column
df = df.drop(columns=['Weighted_Price'])

# Rename the column Timestamp to Date
df = df.rename(columns={'Timestamp': 'Date'})

# Convert the timestamp values to date values
df['Date'] = pd.to_datetime(df['Date'], unit='s')

# Index the data frame on Date
df = df.set_index('Date')

# Missing values in Close should be set to the previous row value
df['Close'] = df['Close'].ffill()

# Missing values in High, Low, Open should be set to the same row's Close value
df['High'] = df['High'].fillna(df['Close'])
df['Low'] = df['Low'].fillna(df['Close'])
df['Open'] = df['Open'].fillna(df['Close'])

# Missing values in Volume_(BTC) and Volume_(Currency) should be set to 0
df['Volume_(BTC)'] = df['Volume_(BTC)'].fillna(0)
df['Volume_(Currency)'] = df['Volume_(Currency)'].fillna(0)

# Plot the data from 2017 and beyond at daily intervals and group the values
# Filter data from 2017 onwards
df_2017_plus = df[df.index >= '2017-01-01']

# Resample to daily intervals and aggregate
daily_data = df_2017_plus.resample('D').agg({
    'High': 'max',
    'Low': 'min',
    'Open': 'mean',
    'Close': 'mean',
    'Volume_(BTC)': 'sum',
    'Volume_(Currency)': 'sum'
})

# Plot all six columns on a single plot
plt.figure(figsize=(10, 6))
plt.plot(daily_data.index, daily_data['High'], label='High')
plt.plot(daily_data.index, daily_data['Low'], label='Low')
plt.plot(daily_data.index, daily_data['Open'], label='Open')
plt.plot(daily_data.index, daily_data['Close'], label='Close')
plt.plot(daily_data.index, daily_data['Volume_(BTC)'], label='Volume_(BTC)')
plt.plot(daily_data.index, daily_data['Volume_(Currency)'],
         label='Volume_(Currency)')
plt.legend()
plt.xlabel('Date')
plt.show()

# Return the transformed DataFrame before plotting
print(daily_data)
