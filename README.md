# Avellaneda-Stoikov Market Making Implementation

This repository implements the optimal market making strategy from "High-frequency trading in a limit order book" by Avellaneda & Stoikov (2008). The implementation includes both data collection and parameter calculation components for automated market making.

## Overview

The Avellaneda-Stoikov model provides a mathematically optimal approach to market making by balancing inventory risk against profit maximization. The strategy dynamically adjusts bid/ask spreads based on:

- Current inventory position
- Market volatility (σ)
- Order arrival intensity (λ)
- Risk aversion parameter (γ)
- Time remaining in trading session

## Files Structure

### 1. `calculate_avellaneda_parameters.py` (Main Implementation)

The core implementation of the Avellaneda-Stoikov market making model. This script performs a complete parameter estimation and optimization workflow.

#### Key Features

- **Volatility Estimation**: Calculates market volatility (σ) using log-returns
- **Order Intensity Fitting**: Models order arrival rates λ(δ) = A × exp(-k × δ)
- **Risk Parameter Optimization**: Finds optimal risk aversion γ via backtesting
- **Real-time Price Calculation**: Generates optimal bid/ask prices
- **Walk-forward Analysis**: Out-of-sample performance evaluation

#### Mathematical Foundation

The model implements these core equations:

```
Reservation Price: r_t = S_t - q_t × γ × σ² × (T-t)
Optimal Spreads: δ±_t = γσ²(T-t)/2 + ln(1+γ/k)/γ ± inventory_adjustment
Bid Price: r_t - δ_b
Ask Price: r_t + δ_a
```

#### Workflow

1. **Data Loading**: Loads historical price and trade data from CSV files
2. **Volatility Calculation**: Estimates σ using daily log-return standard deviations
3. **Intensity Estimation**: Fits exponential decay model to order arrival rates
4. **Gamma Optimization**: Tests risk aversion parameters to maximize risk-adjusted returns
5. **Parameter Output**: Saves optimal parameters and current bid/ask prices to JSON

#### Configuration

```python
TICKER = 'BTC'  # Asset symbol
tick_size = 0.1  # Minimum price increment
limit_order_refresh_rate = '10s'  # Order update frequency
gamma_grid_to_test = np.logspace(np.log10(0.01), np.log10(8.0), 32)  # Risk parameters to test
```

#### Input Data Requirements

The script expects CSV files in the `HL_data/` directory:
- `prices_{TICKER}.csv`: Bid/ask price data with columns: timestamp, price, side
- `trades_{TICKER}.csv`: Trade execution data with columns: timestamp, price, size, side, trade_id

#### Output

- **JSON File**: `avellaneda_parameters_{TICKER}.json` containing:
  - Market data (mid-price, volatility, intensity parameters)
  - Optimal parameters (γ)
  - Current state (inventory, time remaining)
  - Calculated prices (reservation price, bid/ask prices, spreads)

- **Console Output**: Detailed terminal summary with formatted parameters and spreads

#### Usage

```bash
python calculate_avellaneda_parameters.py
```

The script will:
1. Load and process market data
2. Estimate model parameters
3. Optimize risk aversion parameter
4. Output optimal bid/ask prices for current market conditions

### 2. `hyperliquid_data_collector.py` (Data Collection)

A high-performance data collection system for gathering real-time market data from Hyperliquid exchange via WebSocket connections.

#### Features

- **Multi-Symbol Support**: Collect data for multiple assets simultaneously
- **Real-time WebSocket Streams**: Live price, trade, and order book data
- **Robust Connection Management**: Automatic reconnection with exponential backoff
- **CSV Output**: Structured data storage for analysis
- **Performance Monitoring**: Built-in statistics and health monitoring

#### Data Types Collected

1. **Price Data**: Best bid/offer updates
2. **Trade Data**: Executed transactions with price, size, side
3. **Order Book Data**: Full depth order book snapshots
The `calculate_avellaneda_parameters.py` script would ideally need several days of data collected.
#### Configuration

```python
symbols = ['BTC', 'ETH', 'SOL']  # Assets to collect
output_dir = "HL_data"  # Data storage directory
orderbook_depth = 20  # Order book levels to capture
```

#### Output Files

For each symbol, generates:
- `prices_{symbol}.csv`: Best bid/offer data
- `trades_{symbol}.csv`: Trade execution data
- `orderbook_{symbol}.csv`: Order book snapshots

## Installation

### Requirements

```bash
pip install numpy pandas scipy
pip install hyperliquid-python-sdk  # For data collection
```

### Setup

1. Clone/download the repository
2. Install dependencies
3. Create data directory: `mkdir HL_data`
4. Configure ticker symbols in both scripts

## Quick Start

### 1. Collect Data

```bash
python hyperliquid_data_collector.py
```
Only `prices_*.csv` and `trades_*.csv` are used by `calculate_avellaneda_parameters.py`, as it is now. `orderbooks_*.csv` are not used.
The `calculate_avellaneda_parameters.py` script would ideally need several days of data collected.

### 2. Calculate Parameters

```bash
python calculate_avellaneda_parameters.py
```

### 3. Review Output

Check the generated JSON file for optimal parameters:

```json
{
    "ticker": "BTC",
    "market_data": {
        "mid_price": 43250.0,
        "sigma": 0.0124,
        "k": 0.1
    },
    "optimal_parameters": {
        "gamma": 1.2345
    },
    "limit_orders": {
        "ask_price": 43275.50,
        "bid_price": 43224.50,
        "delta_a_percent": 0.059,
        "delta_b_percent": 0.059
    }
}
```

## Model Parameters

### Key Variables

- **σ (sigma)**: Market volatility (annualized)
- **γ (gamma)**: Risk aversion parameter (higher = more conservative)
- **A**: Base order arrival intensity
- **k**: Order intensity decay rate
- **δ**: Spread from mid-price
- **q**: Current inventory position
- **r**: Reservation price (fair value given inventory)

### Parameter Interpretation

- **High γ**: Wider spreads, lower inventory risk, lower profits
- **Low γ**: Tighter spreads, higher inventory risk, higher profits
- **High σ**: Wider spreads due to increased price uncertainty
- **Large |q|**: Asymmetric spreads to encourage inventory reduction

## Risk Management

The model incorporates several risk management mechanisms:

1. **Inventory Penalty**: Positions away from zero incur quadratic costs
2. **Time Decay**: Spreads widen as trading session approaches end
3. **Volatility Adjustment**: Spreads increase with market uncertainty
4. **Dynamic Rebalancing**: Continuous adjustment based on market conditions

## Performance Considerations

- **Walk-Forward Analysis**: Uses historical parameters to trade forward periods
- **Out-of-Sample Testing**: Avoids look-ahead bias in parameter optimization
- **Risk-Adjusted Scoring**: Optimizes Sharpe ratio penalized by final inventory
- **Transaction Costs**: Includes trading fees in PnL calculations

## Limitations

1. **Market Impact**: Assumes orders don't affect market prices
2. **Perfect Execution**: Assumes all limit orders execute at desired prices
3. **Constant Parameters**: Model parameters estimated daily, not intraday
4. **Single Asset**: No cross-asset inventory management
5. **No Adverse Selection**: Assumes no information asymmetry

## Academic Reference

Avellaneda, M., & Stoikov, S. (2008). High-frequency trading in a limit order book. *Quantitative Finance*, 8(3), 217-224.

## License


This implementation is for educational and research purposes. Please ensure compliance with applicable trading regulations and exchange terms of service when using with live data.


