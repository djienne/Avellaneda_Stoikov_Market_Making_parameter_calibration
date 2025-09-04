# Avellaneda-Stoikov Market Making Model Parameter Calculator
# This script implements the optimal market making strategy from 
# "High-frequency trading in a limit order book" by Avellaneda & Stoikov (2008)

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import scipy.optimize
import sys
import os
import numba as nb
from numba import jit
from multiprocessing import Pool
import json

# How frequently limit orders are refreshed (impacts order fill detection)
limit_order_refresh_rate = '10s'

# Set the ticker symbol here
TICKER = 'BTC'  # Change this to any ticker like 'ETH', 'SOL', 'WLFI', 'BTC'.

# Set tick size based on ticker - minimum price increment for the asset
# This affects the granularity of delta values tested in order arrival intensity fitting
if TICKER == 'BTC':
    tick_size = 0.1
elif TICKER == 'ETH':
    tick_size = 0.1  # ETH needs reasonable tick size relative to its price
elif TICKER == 'SOL':
    tick_size = 0.01 
elif TICKER == 'WLFI':
    tick_size = 0.0001  # WLFI prices ~$0.2-0.4, need smaller tick size 
else:
    tick_size = 0.01  # Default for other tickers

# Price deltas (spreads from mid-price) to test for order arrival intensity estimation
# Range from 1 tick to 1000 ticks with 32 points for fitting λ(δ) = A*exp(-k*δ)
delta_list = np.linspace(tick_size, tick_size*1000, 32)

# Risk aversion parameter grid for gamma optimization
# Log-spaced from 0.01 to 8.0 to test wide range of risk preferences
gamma_grid_to_test = np.logspace(np.log10(0.01), np.log10(8.0), 32)

# Load trades data
def load_trades_data(csv_path):
    """
    Load trades data from CSV file into pandas DataFrame.
    
    Parameters:
    csv_path (str): Path to the trades CSV file
    
    Returns:
    pd.DataFrame: DataFrame with trades data including timestamp, price, size, side, trade_id, exchange_timestamp
    """
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Convert timestamp to datetime
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
    
    # Set datetime as index for easier time-based operations
    df = df.set_index('datetime')
    
    return df

def load_and_resample_mid_price(csv_path):
    """
    Load level-1 order book data and calculate mid-price time series.
    Resamples to 1-second intervals for consistent time grid.
    
    Parameters:
    csv_path (str): Path to CSV with bid/ask price data
    
    Returns:
    pd.DataFrame: Time-indexed DataFrame with mid_price, price_bid, price_ask columns
    """
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Convert timestamp to datetime
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')

    # Separate bids and asks
    bids = df[df['side'] == 'bid'].copy()
    asks = df[df['side'] == 'ask'].copy()
    
    # Merge bid and ask data on timestamp to get mid price
    merged = pd.merge(bids, asks, on='datetime', suffixes=('_bid', '_ask'))
    merged['mid_price'] = (merged['price_bid'] + merged['price_ask']) / 2
    
    # Sort by timestamp
    merged = merged.sort_values('datetime')

    print(merged)

    merged = merged.loc[:,['datetime', 'mid_price', 'price_bid', 'price_ask']]

    merged = merged.set_index('datetime')
    
    # Forward-fill to create regular 1-second time grid
    merged = merged.resample('s').ffill()
    
    return merged

# Load mid_price data from CSV
csv_file_path = f'./HL_data/prices_{TICKER}.csv'

# Check if file exists
if not os.path.exists(csv_file_path):
    print(f"Error: File {csv_file_path} not found!")
    print(f"Please ensure you have price data for {TICKER} in the HL_data directory.")
    sys.exit(1)

print(f"Loading data for ticker: {TICKER}")
mid_price_df = load_and_resample_mid_price(csv_file_path)

trades_df = load_trades_data(f'./HL_data/trades_{TICKER}.csv')
trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'],unit='s')
trades_df = trades_df.set_index('timestamp')
trades_df = trades_df.sort_values('timestamp')
# Create buy orders dataframe
buy_trades = trades_df[trades_df['side'] == 'buy'].copy()
# Create sell orders dataframe  
sell_trades = trades_df[trades_df['side'] == 'sell'].copy()

print(mid_price_df)

# Extract unique trading days, removing the last incomplete day
list_of_days = mid_price_df.index.normalize().unique().tolist()[:-1]

print(list_of_days)

print(len(mid_price_df))
print(len(list_of_days))

# ------------------------------------------------------------- CALCULATION OF sigma -------------------------------------------------------------
# Volatility estimation using log-returns on daily basis
# σ represents the diffusion coefficient in the Avellaneda-Stoikov model: dS_t = σS_t dW_t

# Calculate daily volatility from log-returns
std = (np.log(mid_price_df.loc[:,'mid_price']).diff()  # Log-return differences
                   .dropna()  # Remove any NaNs
                   .groupby(pd.Grouper(freq='1D')).std()  # Daily standard deviation
                   .rolling('2D').mean())  # 2-day rolling average for smoothing

# Annualize volatility (multiply by sqrt of seconds per day)
sigma_list = std * np.sqrt(60*60*24)
sigma_list = sigma_list.tolist()

sigma_list = sigma_list[:-1]  # remove last day because it is incomplete

print(f"sigma_list: {sigma_list}")
print(f"len(sigma_list): {len(sigma_list)}")

# ------------------------------------------------------------- CALCULATION OF A and k -------------------------------------------------------------
# Order arrival intensity estimation: λ(δ) = A * exp(-k * δ)
# Where δ is the distance from mid-price and λ is the arrival rate of market orders
# A represents the base intensity, k represents the exponential decay rate

# Reference mid-price for initial calculations
first_mid_price_value = mid_price_df.loc[:,'mid_price'].dropna().iloc[0]

print(delta_list)

# Exponential decay function for fitting order arrival intensity
# This models how market order arrival rate decreases with distance from mid-price
def exp_fit(x,a,b):
    """Exponential decay function: y = a * exp(-b * x)"""
    y = a*np.exp(-b*x)
    return y

Alist = []
klist = []

print(f"Doing {len(list_of_days)} days")

def process_dates(list_of_dates, buy_orders, sell_orders, deltalist, exp_fit):
    """
    Process dates using real buy and sell order data instead of mid-price movements.
    
    Parameters:
    - list_of_dates: List of dates to process
    - buy_orders: DataFrame with buy orders (price, size) indexed by datetime
    - sell_orders: DataFrame with sell orders (price, size) indexed by datetime  
    - deltalist: List of price deltas to test
    - exp_fit: Exponential fitting function
    """
    Alist, klist = [], []

    for i in range(len(list_of_dates)):
        day = list_of_dates[i].strftime('%Y-%m-%d')

        # Extract day's buy and sell orders
        day_buy_orders = buy_orders.loc[day].copy() if day in buy_orders.index.strftime('%Y-%m-%d') else pd.DataFrame()
        day_sell_orders = sell_orders.loc[day].copy() if day in sell_orders.index.strftime('%Y-%m-%d') else pd.DataFrame()
        
        # Skip if no orders for this day
        if day_buy_orders.empty and day_sell_orders.empty:
            continue

        best_bid = day_buy_orders['price'].max()
        best_ask = day_sell_orders['price'].min()
        reference_mid = (best_bid + best_ask) / 2

        deltadict = {}

        for price_delta in deltalist:
            # Define limit order levels
            limit_bid = reference_mid - price_delta  # Limit buy order
            limit_ask = reference_mid + price_delta  # Limit sell order
            
            # Find when limit orders would be hit
            bid_hits = []  # When sell orders hit our limit buy
            ask_hits = []  # When buy orders hit our limit sell
            
            # Check sell orders against our limit buy order
            if not day_sell_orders.empty:
                sell_hits_bid = day_sell_orders[day_sell_orders['price'] <= limit_bid]
                if not sell_hits_bid.empty:
                    bid_hits = sell_hits_bid.index.tolist()
            
            # Check buy orders against our limit sell order  
            if not day_buy_orders.empty:
                buy_hits_ask = day_buy_orders[day_buy_orders['price'] >= limit_ask]
                if not buy_hits_ask.empty:
                    ask_hits = buy_hits_ask.index.tolist()
            
            # Combine all hit times
            all_hits = sorted(bid_hits + ask_hits)
            
            if len(all_hits) > 1:
                # Calculate time differences between hits
                hit_times = pd.DatetimeIndex(all_hits)
                deltas = hit_times.to_series().diff().dt.total_seconds().dropna()
                deltadict[price_delta] = deltas
            else:
                # If no hits or only one hit, use a large default value
                deltadict[price_delta] = pd.Series([86400])  # 24 hours in seconds

        # Build lambda dataframe (arrival rate = 1/mean_time_between_hits)
        # λ(δ) represents how frequently market orders arrive at distance δ from mid-price
        lambdas = pd.DataFrame({
            "delta": list(deltadict.keys()),
            "lambda_delta": [1 / d.mean() if len(d) > 0 else 1e-6 for d in deltadict.values()]
        }).set_index("delta")

        # Fit exponential decay model λ(δ) = A * exp(-k * δ) to empirical arrival rates
        try:
            paramsB, _ = scipy.optimize.curve_fit(
                exp_fit,
                lambdas.index.values,
                lambdas["lambda_delta"].values,
                maxfev=5000  # Increase max iterations for robustness
            )
            A, k = paramsB
            Alist.append(A)
            klist.append(k)
        except (RuntimeError, ValueError):
            # If fitting fails, append NaN values
            Alist.append(float('nan'))
            klist.append(float('nan'))

    return Alist, klist

Alist, klist = process_dates(list_of_days, buy_trades, sell_trades, delta_list, exp_fit)

print(f"Alist: {Alist}")
print(f"klist: {klist}")
# ------------------------------------------------------------- GAMMA OPTIMIZATION AND BACKTESTING -------------------------------------------------------------
# Find optimal risk aversion parameter γ by maximizing risk-adjusted PnL
# γ controls the trade-off between inventory risk and profit maximization

def run_backtest(mid_prices, buy_trades, sell_trades, gamma, k, sigma, fee=0.00030):
    """
    Simulate Avellaneda-Stoikov market making strategy with given parameters.
    
    The strategy sets bid/ask prices based on:
    - Reservation price: r_t = S_t - q_t * γ * σ² * (T-t)
    - Optimal spreads: δ^±_t = γ*σ²*(T-t)/2 + ln(1+γ/k)/γ ± skew_adjustment
    
    Parameters:
    mid_prices: Time series of mid-prices
    buy_trades, sell_trades: Market order data for fill simulation
    gamma: Risk aversion parameter
    k: Order intensity decay parameter  
    sigma: Volatility parameter
    fee: Trading fee rate
    
    Returns:
    dict: Backtest results including PnL, inventory, spreads, and order prices
    """
    # Pre-process orders into time-aligned series for faster lookup
    time_index = mid_prices.index
    
    # Create boolean series indicating when orders hit price levels
    # This is more efficient for repeated backtesting
    # Handle duplicate indices by keeping the last value for each timestamp
    # num_duplicates = buy_trades.index.duplicated().sum()
    # print(f"Number of duplicate timestamps: {num_duplicates}")

    buy_trades_clean = buy_trades.groupby(level=0).min()
    sell_trades_clean = sell_trades.groupby(level=0).max()
    
    # Get min buy price and max sell price in each time window
    buy_min = buy_trades_clean['price'].resample(limit_order_refresh_rate).min().reindex(time_index, method='ffill')
    sell_max = sell_trades_clean['price'].resample(limit_order_refresh_rate).max().reindex(time_index, method='ffill')

    # Simulate order refresh lag - cannot update orders faster than limit_order_refresh_rate
    mid_prices = mid_prices.resample(limit_order_refresh_rate).first().reindex(time_index, method='ffill')
    
    N = len(time_index)
    T = 1.0  # Normalized trading horizon (1 day)
    dt = T/N  # Time step
    
    s_values = mid_prices.values
    time_remaining = T - np.arange(len(s_values)) * dt  # Time to end of trading period
    
    # Initialize state variables
    q = np.zeros(len(s_values) + 1)  # Inventory (number of shares held)
    x = np.zeros(len(s_values) + 1)  # Cash position
    
    # Calculate optimal spread using Avellaneda-Stoikov formula
    # δ^± = γσ²(T-t) + (2/γ)ln(1+γ/k) ± inventory_adjustment
    spread_base = gamma * sigma**2.0 * time_remaining + (2.0 / gamma) * np.log(1.0 + (gamma / k))
    half_spread = spread_base / 2.0
    
    # Initialize tracking arrays
    pnl = np.zeros(len(s_values) + 1)  # Profit and loss over time
    spr = np.zeros(len(s_values) + 1)  # Spread values over time
    r = np.zeros(len(s_values) + 1)    # Reservation prices over time
    r_a = np.zeros(len(s_values) + 1)  # Ask prices over time
    r_b = np.zeros(len(s_values) + 1)  # Bid prices over time
    
    for i in range(N):
        # Calculate reservation price: r_t = S_t - q_t * γ * σ² * (T-t)
        # This represents the "fair value" given current inventory and remaining time
        r[i] = s_values[i] - q[i] * gamma * sigma**2 * time_remaining[i]
        spr[i] = spread_base[i]
        
        # Calculate inventory adjustment (gap between reservation price and mid-price)
        gap = abs(r[i] - s_values[i])
        
        # Adjust spreads based on inventory position
        # If reservation price > mid-price (short inventory), widen ask and tighten bid
        # If reservation price < mid-price (long inventory), tighten ask and widen bid
        if r[i] >= s_values[i]:
            delta_a = half_spread[i] + gap  # Widen ask spread
            delta_b = half_spread[i] - gap  # Tighten bid spread
        else:
            delta_a = half_spread[i] - gap  # Tighten ask spread
            delta_b = half_spread[i] + gap  # Widen bid spread
        
        # Calculate actual bid and ask prices
        r_a[i] = r[i] + delta_a  # Ask price
        r_b[i] = r[i] - delta_b  # Bid price
        
        # Check if market orders hit our limit orders
        # sell=1 if market sell order hits our ask (we buy)
        # buy=1 if market buy order hits our bid (we sell)
        sell = 1 if pd.notna(sell_max.iloc[i]) and sell_max.iloc[i] >= r_a[i] else 0
        buy = 1 if pd.notna(buy_min.iloc[i]) and buy_min.iloc[i] <= r_b[i] else 0
        
        # Update inventory: +1 when we buy (market sell hits our ask), -1 when we sell (market buy hits our bid)
        q[i+1] = q[i] + (sell - buy)
        # Calculate cash flows from trading (including fees)
        if sell:
            sell_proceeds = r_a[i]  # Revenue from selling at ask price
            sell_fee = sell_proceeds * fee
            sell_net = sell_proceeds - sell_fee  # Net proceeds after fees
        else:
            sell_net = 0
            
        if buy:
            buy_cost = r_b[i]  # Cost of buying at bid price
            buy_fee = buy_cost * fee
            buy_total = buy_cost + buy_fee  # Total cost including fees
        else:
            buy_total = 0
        
        # Update cash position and mark-to-market PnL
        x[i+1] = x[i] + sell_net - buy_total  # Cash balance
        pnl[i+1] = x[i+1] + q[i+1] * s_values[i]  # Total PnL (cash + inventory value)
    
    return {
        'pnl': pnl, 
        'x': x, 
        'q': q, 
        'spread': spr, 
        'r': r, 
        'r_a': r_a, 
        'r_b': r_b
    }


gammalist = []

print(gamma_grid_to_test)

# Walk-forward analysis: use parameters estimated from day t-1 to trade on day t
# This simulates realistic out-of-sample performance
for j in range(1, len(list_of_days)):

    # Get parameters from previous day
    sigma = sigma_list[j-1]
    A = Alist[j-1]
    k = klist[j-1]

    date = list_of_days[j].strftime('%Y-%m-%d')
    print(date)

    # Get mid prices for the day
    s = mid_price_df.loc[date]

    # Fill in any missing seconds with NaNs
    s = s.resample('s').asfreq(fill_value=np.nan)

    # Forward fill the previous midprice if one doesn't
    # exist for a given second-interval
    s = s.ffill()
    s = s['mid_price']

    m = mid_price_df.loc[date]

    def backtest_gamma(gamma, s, k, sigma):
        """Test a specific gamma value and return risk-adjusted performance score"""
        # Run the backtest with current gamma
        res = run_backtest(s, buy_trades.loc[date], sell_trades.loc[date], gamma, k, sigma)

        # Calculate PnL returns for Sharpe ratio calculation
        res_pnl = pd.Series(res['pnl']).pct_change().dropna()[1:]

        final_pnl = res['pnl'][-1]
        print("-"*20)

        print(f"final_pnl: {final_pnl}")

        print(f"Final inventory: {res['q'][-1]}")
        
        # Calculate Sharpe ratio (risk-adjusted return)
        if len(res_pnl) == 0 or res_pnl.std() == 0:
            sharpe = float('nan')
        else:
            sharpe = res_pnl.mean() / res_pnl.std()

        print(f"Sharpe: {sharpe}")
        
        # Score penalizes large final inventory (inventory risk)
        # Higher Sharpe with lower absolute inventory is preferred
        score = sharpe / abs(res['q'][-1]) * 10000.0 if res['q'][-1] != 0 else sharpe * 10000.0
        print(f"score: {score}")

        return [round(gamma,5), score]

    # Test different gamma values sequentially
    gamma_results = []
    for gamma in gamma_grid_to_test:
        print("-"*20)
        print(f"Doing gamma: {gamma}")
        result = backtest_gamma(gamma, s, k, sigma)
        
        gamma_results.append(result)

    # Find the best gamma, excluding NaN and zero PnL results
    results_df = pd.DataFrame(gamma_results, columns=['gamma', 'score'])
    valid_results = results_df[(results_df['score'] != 0) & (~results_df['score'].isna())]

    valid_results = valid_results.set_index('gamma')

    print("-"*20)
    print(valid_results.index)
    print("-"*20)
    print(valid_results)
        
    if len(valid_results) > 0:
        best_gamma = valid_results['score'].idxmax()
    else:
        # Fallback to a reasonable gamma if no valid results
        best_gamma = 0.5
        
    print("-"*20)
    print(f"best gamma: {best_gamma}")
    print(f"best score: {valid_results.loc[best_gamma]['score']}")
    print("-"*20)
    
    gammalist.append(best_gamma)

print(gammalist)

# ------------------------------------------------------------- FINAL PARAMETER CALCULATION -------------------------------------------------------------
# Calculate current optimal bid/ask prices using latest parameters and current market state

s = mid_price_df.loc[:,'mid_price'].iloc[-2]  # Latest mid-price
gamma = gammalist[-2]  # Optimal risk aversion parameter
sigma = sigma_list[-2]  # Latest volatility estimate
k = klist[-2]  # Latest order intensity decay parameter
time_remaining = 0.16666  # Fraction of trading day remaining (e.g. at 20:00 it is 0.16666)
q = 10.0  # Current inventory position (positive = long, negative = short)

# Calculate base spread using Avellaneda-Stoikov formula
spread_base = gamma * sigma**2.0 * time_remaining + (2.0 / gamma) * np.log(1.0 + (gamma / k))
half_spread = spread_base / 2.0

# Calculate inventory penalty
print(q * gamma * sigma**2.0 * time_remaining)

# Calculate reservation price (our "fair value" given inventory and time remaining)
r = s - q * gamma * sigma**2.0 * time_remaining

# Calculate gap between reservation price and current mid-price
gap = abs(r - s)

# Apply inventory adjustment to spreads
# When long (positive inventory): tighten ask to sell faster, widen bid to buy slower
# When short (negative inventory): widen ask to sell slower, tighten bid to buy faster
if r >= s:  # Reservation price above mid (short inventory, want to buy)
    delta_a = half_spread + gap  # Widen ask (sell at higher price)
    delta_b = half_spread - gap  # Tighten bid (buy more aggressively)
else:       # Reservation price below mid (long inventory, want to sell)
    delta_a = half_spread - gap  # Tighten ask (sell more aggressively)
    delta_b = half_spread + gap  # Widen bid (buy at lower price)

# Calculate limit order prices
r_a = r + delta_a
r_b = r - delta_b

# Calculate relative percentages
delta_a_percent = (delta_a / s) * 100.0
delta_b_percent = (delta_b / s) * 100.0

# Create results dictionary
results = {
    "ticker": TICKER,
    "timestamp": pd.Timestamp.now().isoformat(),
    "market_data": {
        "mid_price": float(s),
        "sigma": float(sigma),
        "A": float(Alist[-1]) if len(Alist) > 0 else None,
        "k": float(k)
    },
    "optimal_parameters": {
        "gamma": float(gamma)
    },
    "current_state": {
        "time_remaining": float(time_remaining),
        "inventory": int(q)
    },
    "calculated_values": {
        "reservation_price": float(r),
        "gap": float(gap),
        "spread_base": float(spread_base),
        "half_spread": float(half_spread)
    },
    "limit_orders": {
        "ask_price": float(r_a),
        "bid_price": float(r_b),
        "delta_a": float(delta_a),
        "delta_b": float(delta_b),
        "delta_a_percent": float(delta_a_percent),
        "delta_b_percent": float(delta_b_percent)
    }
}

# Save to JSON file
json_filename = f"avellaneda_parameters_{TICKER}.json"
with open(json_filename, 'w') as f:
    json.dump(results, f, indent=4)

# Print nice terminal summary
print("\n" + "="*80)
print(f"AVELLANEDA-STOIKOV MARKET MAKING PARAMETERS - {TICKER}")
print("="*80)
print(f"Market Data:")
print(f"   Mid Price:           ${s:,.4f}")
print(f"   Volatility (sigma):  {sigma:.6f}")
print(f"   Intensity (A):       {Alist[-1]:.4f}" if len(Alist) > 0 else "   Intensity (A):       N/A")
print(f"   Decay (k):           {k:.6f}")
print(f"\nOptimal Parameters:")
print(f"   Risk Aversion (gamma): {gamma:.6f}")
print(f"\nCurrent State:")
print(f"   Time Remaining:        {time_remaining:.4f}")
print(f"   Inventory (q):         {q}")
print(f"\nCalculated Prices:")
print(f"   Mid Price:             ${s:.4f}")
print(f"   Reservation Price:     ${r:.4f}")
print(f"   Ask Price:             ${r_a:.4f}")
print(f"   Bid Price:             ${r_b:.4f}")
print(f"\nSpreads:")
print(f"   Delta Ask:             ${delta_a:.4f} ({delta_a_percent:.4f}%)")
print(f"   Delta Bid:             ${delta_b:.4f} ({delta_b_percent:.4f}%)")
print(f"   Total Spread:          {(delta_a_percent + delta_b_percent):.4f}%")
print(f"\nResults saved to: {json_filename}")
print("="*80)