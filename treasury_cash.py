import json
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
import statsmodels.api as sm
from matplotlib.dates import DateFormatter
from sklearn.linear_model import TheilSenRegressor

# Fetch data from the API
response = requests.get(
    "https://api.fiscaldata.treasury.gov/services/api/fiscal_service/v1/accounting/dts/dts_table_1?filter=record_date:gt:2023-04-18")
json_data = json.loads(response.text)

# Load data into a DataFrame
df = pd.DataFrame(json_data['data'])

# Convert 'record_date' to datetime and 'open_today_bal' to float
df['record_date'] = pd.to_datetime(df['record_date'])
df['open_today_bal'] = df['open_today_bal'].astype(float)

# Filter for the relevant account types
df_tga = df[df['account_type'] ==
            'Treasury General Account (TGA) Opening Balance']
df_deposits = df[df['account_type'] == 'Total TGA Deposits (Table II)']
df_withdrawals = df[df['account_type'] ==
                    'Total TGA Withdrawals (Table II) (-)']

# Seaborn settings
sns.set_theme(style="darkgrid")

# Create the plot
fig, ax1 = plt.subplots(figsize=(8, 6))

# Plot the TGA Opening Balance
sns.lineplot(
    data=df_tga,
    x='record_date',
    y='open_today_bal',
    ax=ax1,
    color='g',
    label='Opening Balance')

# Add deposits and withdrawals as bars
ax1.bar(
    df_deposits['record_date'],
    df_deposits['open_today_bal'],
    color='green',
    label='Deposits')
ax1.bar(df_withdrawals['record_date'], -
        df_withdrawals['open_today_bal'], color='red', label='Withdrawals')

# Set the title and labels
ax1.set_title(
    'Countdown to Economic Catastrophe: US Treasury Account Movements',
    fontsize=14)
ax1.set_xlabel('Date', fontsize=12)
ax1.set_ylabel('Balance', fontsize=12)


# Assume df_tga is your DataFrame and it has two columns: 'record_date'
# and 'open_today_bal'
df_tga['record_date_numeric'] = (
    df_tga['record_date'] -
    df_tga['record_date'].min()).dt.days

X = df_tga['record_date_numeric'].values.reshape(-1, 1)
y = df_tga['open_today_bal'].values

# Fit the Theil-Sen Regression model
reg = TheilSenRegressor().fit(X, y)

# Predict when the balance will reach zero by solving the equation y = mx
# + b for x when y = 0
zero_balance_day_numeric = int(-reg.intercept_ / reg.coef_[0])

# Convert the numeric day back to a date
zero_balance_date = df_tga['record_date'].min(
) + timedelta(days=zero_balance_day_numeric)

print(
    f"Based on the Theil-Sen Regression model, the balance will reach zero on {zero_balance_date.strftime('%Y-%m-%d')}.")

# Add the text at the bottom of the chart
plt.text(
    0.5,
    0.01,
    f"Based on the model, the balance will reach zero on {zero_balance_date.strftime('%Y-%m-%d')}.",
    horizontalalignment='center',
    verticalalignment='bottom',
    transform=ax1.transAxes)

# Format date labels
date_form = DateFormatter("%m-%d")
ax1.xaxis.set_major_formatter(date_form)

# Rotate and align the date labels
fig.autofmt_xdate()

plt.legend(loc='upper left')
plt.show()
