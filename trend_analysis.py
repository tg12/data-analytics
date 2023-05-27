import datetime as dt

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tabulate import tabulate

import pandas as pd


def detect_rangebound(data):
    """
    Detects if a data series is range-bound based on multiple checks.

    Parameters:
        data (array-like): Input data series.

    Returns:
        bool: True if the data series is range-bound, False otherwise.
    """
    # Check 1: Calculate the variance of the data
    variance = np.var(data)

    # Check 2: Calculate the range of the data
    data_range = np.max(data) - np.min(data)

    # Check 3: Calculate the standard deviation of the data
    std_dev = np.std(data)

    # Calculate the dynamic variance threshold based on the data
    # characteristics
    threshold = 0.01 * data_range + 0.1 * std_dev

    print("Variance:", variance)
    print("Data Range:", data_range)
    print("Standard Deviation:", std_dev)
    print("Threshold:", threshold)

    # Determine if the data series is range-bound based on the variance
    # threshold
    if variance <= threshold:
        return True
    else:
        return False


def process_data(data):
    # Calculate variance of the data
    variance = np.var(data)

    # Calculate the mean and standard deviation
    mean = np.mean(data)
    std_dev = np.std(data)

    # Find outliers based on a threshold of 10 standard deviations
    threshold = 10 * std_dev
    outliers = [x for x in data if abs(x - mean) > threshold]

    # Replace outliers with the last known good value
    cleaned_data = []
    last_good_value = None
    for value in data:
        if value in outliers:
            if last_good_value is not None:
                cleaned_data.append(last_good_value)
        else:
            cleaned_data.append(value)
            last_good_value = value

    # Calculate the percentage of values deviating by more than 10 standard
    # deviations
    deviation_percentage = (len(outliers) / len(data)) * 100

    # Find the 5 lowest and highest samples
    sorted_data = sorted(cleaned_data)
    lowest_samples = sorted_data[:5]
    highest_samples = sorted_data[-5:]

    return cleaned_data, deviation_percentage, lowest_samples, highest_samples


def detect_trend(cleaned_data):
    # Calculate the differences between consecutive values
    differences = np.diff(cleaned_data)

    # Count the number of positive and negative differences
    num_positive_diff = sum(diff > 0 for diff in differences)
    num_negative_diff = sum(diff < 0 for diff in differences)

    # Calculate the weights based on the difference counts and time series
    # length
    total_diff = num_positive_diff + num_negative_diff
    time_series_length = len(cleaned_data)
    positive_weight = num_positive_diff / time_series_length
    negative_weight = num_negative_diff / time_series_length

    # Determine the trend based on weighted difference counts
    if positive_weight > negative_weight:
        return "Upward trend (Strong)"
    elif positive_weight < negative_weight:
        return "Downward trend (Strong)"
    else:
        return "Sideways trend"


def calculate_ema(data, window):
    return data.ewm(span=window, adjust=False).mean()


def calculate_signal(macd, signal_window=9):
    return calculate_hull_moving_average(macd, signal_window)


def calculate_hull_moving_average(data, window):
    half_window = int(window / 2)
    wma1 = data.rolling(window=half_window).mean()
    wma2 = data.rolling(window=window).mean()
    diff = (2 * wma1) - wma2
    hull = diff.rolling(window=int(np.sqrt(window))).mean()
    return hull


def calculate_macd(
        data,
        short_window=None,
        long_window=None,
        signal_window=None):
    if short_window is None:
        short_window = int(len(data) / 6)
    if long_window is None:
        long_window = int(len(data) / 4)
    if signal_window is None:
        signal_window = int(len(data) / 10)

    ema_short = calculate_hull_moving_average(data["price"], short_window)
    ema_long = calculate_hull_moving_average(data["price"], long_window)
    macd = ema_short - ema_long
    signal = calculate_hull_moving_average(macd, signal_window)
    df = pd.DataFrame({"macd": macd, "signal": signal}, index=data.index)
    return df


def plot_macd(stock_data, macd, signal, cleaned_data):
    sns.set_style("whitegrid")

    # Combine MACD and Signal into a single DataFrame
    df = pd.concat([macd, signal], axis=1)

    fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(12, 10), sharex=True)

    ax1.plot(stock_data.index, stock_data["price"], label="Close Price")
    ax1.set_title("Stock Close Price")
    ax1.set_ylabel("Price")
    ax1.legend(loc="upper left")

    ax2.plot(df.index, df["macd"], label="MACD", color="blue", linewidth=1.5)
    ax2.plot(
        df.index,
        df["signal"],
        label="Signal",
        color="red",
        linewidth=1.5)
    ax2.set_title("MACD & Signal")
    ax2.set_ylabel("Value")
    ax2.legend(loc="upper left")

    # Plot up and down arrows for trend points
    crossover_buy = np.where(df["macd"] > df["signal"])
    crossover_sell = np.where(df["macd"] < df["signal"])

    for idx in range(1, len(crossover_buy[0])):
        if crossover_buy[0][idx - 1] + 1 == crossover_buy[0][idx]:
            ax2.annotate(
                "↑",
                xy=(
                    df.index[crossover_buy[0][idx]],
                    df["macd"].iloc[crossover_buy[0][idx]],
                ),
                xytext=(-15, 10),
                textcoords="offset points",
                arrowprops=dict(facecolor="g", arrowstyle="->", alpha=0.5),
            )

    for idx in range(1, len(crossover_sell[0])):
        if crossover_sell[0][idx - 1] + 1 == crossover_sell[0][idx]:
            ax2.annotate(
                "↓",
                xy=(
                    df.index[crossover_sell[0][idx]],
                    df["macd"].iloc[crossover_sell[0][idx]],
                ),
                xytext=(-15, -15),
                textcoords="offset points",
                arrowprops=dict(facecolor="r", arrowstyle="->", alpha=0.5),
            )

    # Determine the latest trend
    latest_macd = df["macd"].iloc[-1]
    latest_signal = df["signal"].iloc[-1]

    if latest_macd > latest_signal:
        latest_trend = "up"
    elif latest_macd < latest_signal:
        latest_trend = "down"
    else:
        latest_trend = "neutral"

    # Plot up and down arrows for trend points on cleaned data
    trend_points = [
        detect_trend(cleaned_data[i - 1: i + 2])
        for i in range(1, len(cleaned_data) - 1)
    ]
    buy_indices = [
        i + 1
        for i, trend in enumerate(trend_points)
        if trend == "Upward trend (Strong)"
    ]
    sell_indices = [
        i + 1
        for i, trend in enumerate(trend_points)
        if trend == "Downward trend (Strong)"
    ]

    for idx in buy_indices:
        ax3.annotate(
            "↑",
            xy=(stock_data.index[idx], cleaned_data[idx]),
            xytext=(-15, 10),
            textcoords="offset points",
            arrowprops=dict(facecolor="g", arrowstyle="->", alpha=0.5),
        )

    for idx in sell_indices:
        ax3.annotate(
            "↓",
            xy=(stock_data.index[idx], cleaned_data[idx]),
            xytext=(-15, -15),
            textcoords="offset points",
            arrowprops=dict(facecolor="r", arrowstyle="->", alpha=0.5),
        )

    ax3.plot(
        stock_data.index,
        cleaned_data,
        label="Cleaned Data",
        color="purple")
    ax3.set_title("Cleaned Data")
    ax3.set_ylabel("Value")
    ax3.legend(loc="upper left")

    # Rotate the x-axis dates for better readability
    fig.autofmt_xdate()

    plt.tight_layout()
    plt.show()

    # save the plot to a file

    fig.savefig("Fig_1.png")


def read_csv_data(file_name):
    # first column is date
    # second column is price
    # they both exist in the csv file
    data = pd.read_csv(file_name, index_col=0, parse_dates=True)
    return data


# Read CSV data
csv_file = "backtest_prices.csv"
stock_data = read_csv_data(csv_file)

print(tabulate(stock_data, headers="keys", tablefmt="psql"))

# Calculate MACD and Signal values
macd_dict = calculate_macd(stock_data)
macd = macd_dict["macd"]
signal = macd_dict["signal"]


# Add MACD and Signal columns to the dataframe
stock_data["macd"] = macd
stock_data["signal"] = signal


# Plot the MACD, Signal, and Stock Close Price
cleaned_data, _, _, _ = process_data(stock_data["price"].values)
plot_macd(stock_data, macd, signal, cleaned_data)

is_rangebound = detect_rangebound(stock_data["price"].values)
print("Is Range-Bound:", is_rangebound)


# Calculate the latest trend based on MACD and Signal
latest_trend = detect_trend(cleaned_data)

# Calculate the latest MACD and Signal values
latest_macd = stock_data["macd"].iloc[-1]
latest_signal = stock_data["signal"].iloc[-1]

# Determine if the trend is approaching a reversal
macd_max = stock_data["macd"].max()
signal_max = stock_data["signal"].max()
approaching_reversal = (latest_macd >= 0.8 * macd_max) or (
    latest_signal >= 0.8 * signal_max
)

# Print the latest trend and trade decision

print("Latest trend: {}".format(latest_trend))
print("Approaching reversal: {}".format(approaching_reversal))


# Trade decision
if latest_trend == "Upward trend (Strong)" and approaching_reversal:
    print("Trade Decision: Sell")
elif latest_trend == "Downward trend (Strong)" and approaching_reversal:
    print("Trade Decision: Buy")
elif latest_trend == "Upward trend (Strong)" and not approaching_reversal:
    print("Trade Decision: Buy")
else:
    print("Trade Decision: Hold")
