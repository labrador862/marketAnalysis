compute_rsi() - Relative Strength Index (RSI) and Wilder Smoothing Explained

RSI is a bounded momentum oscillator (0-100) that measures the relative strength of recent price gains versus losses, where:
RSI ~70 or greater -> strong positive momentum
RSI ~50 -> neutral momentum
RSI ~30 or lower -> strong negative momentum

Variable explanations

delta - One day price (close) change: current day - previous
gain/loss - Absolute value of positive or negative price changes
avg_gain/avg_loss - Wilder Smoothing; moving window average over a period (14 days here)
rs - Relative strength, simply avg_gain divided by avg_loss
rsi_14 - Computes final RSI value over a 14 day period