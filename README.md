# Monte Carlo Asian Option Pricer

A compact Monte Carlo pricer for **Asian call options** using historical stock data (via `yfinance`).  
Includes a CLI for single-ticker and multi-ticker comparison modes, convergence CSV exports, and plotting.

## Features
- Monte Carlo path simulation for Asian call payoff (vectorized per-path increments).
- Confidence intervals and convergence checkpointing (CSV export).
- Fetches historical prices and computes annualized volatility from `yfinance`.
- CLI modes:
  - `single` — price one ticker (default SPY)
  - `compare` — price multiple tickers and show comparative bar chart
- Optional interactive demo via Streamlit.

## Quick start

### 1. Install
```bash
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
