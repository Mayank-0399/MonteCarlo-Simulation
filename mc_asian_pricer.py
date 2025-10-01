#!/usr/bin/env python3
"""
Monte Carlo Asian Option Pricer (single vs compare)

Usage:
    python asian_montecarlo.py single [SPY]
    python asian_montecarlo.py compare AAPL MSFT GOOG --simulations 20000 --steps 100

Options:
    --simulations N   Number of Monte Carlo simulations (default: 20000 for compare, 50000 for single)
    --steps M         Number of time steps per path (default: 100)
    --seed S          RNG seed for reproducibility (optional)
    --outdir DIR      Output directory for CSV/plots (default: .)
"""
import argparse
import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from tabulate import tabulate

# --- Monte Carlo Asian Call Pricer ---
def monte_carlo_asian_call(S0, K, T, sigma, r, steps, simulations, seed=None, checkpoint=1000):
    """
    Returns:
      discounted_price (float),
      (conf_low, conf_high) (tuple),
      convergence (list of [simulations_done, discounted_estimate])
    """
    if seed is not None:
        np.random.seed(int(seed))

    dt = float(T) / float(steps)
    payoffs = np.empty(simulations, dtype=float)  # pre-allocate for speed
    convergence = []

    # simulate
    for i in range(simulations):
        St = float(S0)
        # use vectorized increments for each path (faster than inner python loop)
        z = np.random.normal(size=steps)
        increments = np.exp((r - 0.5 * sigma * sigma) * dt + sigma * np.sqrt(dt) * z)
        # Calculate the entire path
        path = St * np.cumprod(increments)
        # The price should be included in the average. The path has 'steps' prices.
        # To include S0, we need steps+1 prices, but common GBM implementation uses 
        # path as S_dt, S_2dt, ..., S_T. We assume the average is over these 'steps' end-of-interval prices.
        avg_price = path.mean() 
        payoff = max(avg_price - float(K), 0.0)
        payoffs[i] = payoff

        # checkpoint convergence every `checkpoint` simulations (and at the end)
        if (i + 1) % checkpoint == 0:
            # Note: math.exp is faster than numpy.exp for single values
            discounted = math.exp(-r * T) * payoffs[:(i + 1)].mean()
            convergence.append([i + 1, discounted])

    # ensure final convergence point included
    discounted_price = math.exp(-r * T) * float(payoffs.mean())
    if not convergence or convergence[-1][0] != simulations:
        convergence.append([simulations, discounted_price])

    # standard error: discount factor multiplies mean and std
    # CORRECTION APPLIED: Use ddof=1 for sample standard deviation (unbiased estimator of population std)
    std_payoffs = float(payoffs.std(ddof=1)) 
    stderr = math.exp(-r * T) * (std_payoffs / math.sqrt(simulations))
    # 1.96 standard deviations for a 95% confidence interval
    conf_low = discounted_price - 1.96 * stderr
    conf_high = discounted_price + 1.96 * stderr

    return discounted_price, (conf_low, conf_high), convergence

# --- Fetch real stock data and implied parameters ---
def fetch_stock_params(ticker, lookback_period_years=5):
    """
    Downloads close price history and returns:
      S0 (last close), K (default strike set as 95% of S0), sigma (annualized vol)
    Raises ValueError if data insufficient.
    """
    # yfinance period strings: "5y" for 5 years - keep simple
    period = f"{int(lookback_period_years)}y"
    try:
        data = yf.download(ticker, period=period, progress=False, auto_adjust=True)
    except Exception as e:
        raise ValueError(f"yfinance error for {ticker}: {e}")

    if data is None or "Close" not in data.columns or data["Close"].dropna().empty:
        raise ValueError(f"No close price data for {ticker} (period={period})")

    closes = data["Close"].dropna()
    if len(closes) < 2:
        raise ValueError(f"Not enough price data for {ticker}")

    S0 = float(closes.iloc[-1])
    # log returns
    # Note: Using ddof=0 for log_returns.std() is common in finance for historical volatility (population vol)
    log_returns = np.log(closes / closes.shift(1)).dropna()
    # annualize using 252 trading days
    sigma = float(log_returns.std(ddof=0) * np.sqrt(252.0))
    K = float(S0 * 0.95)  # default strike slightly below current price
    return S0, K, sigma

# --- Single Asset Mode ---
def run_single(ticker="SPY", T=0.75, r=0.01, steps=100, simulations=50000, seed=None, outdir="."):
    print(f"Running SINGLE mode for {ticker} (T={T}y, steps={steps}, sims={simulations})")
    
    try:
        S0, K, sigma = fetch_stock_params(ticker)
    except ValueError as e:
        print(f"FATAL ERROR: {e}")
        return

    price, ci, convergence = monte_carlo_asian_call(S0, K, T, sigma, r, steps, simulations, seed=seed)

    print(f"\n--- Pricing {ticker} Asian Call Option (K={K:.2f}, T={T} years) ---")
    print(f"S0: {S0:.2f}, Annual Volatility (sigma): {sigma:.4f}, Risk-Free Rate (r): {r}")
    print(f"Price (Asian Call): ${price:.4f}")
    print(f"95% Confidence Interval: [${ci[0]:.4f}, ${ci[1]:.4f}]")

    # export convergence CSV
    df = pd.DataFrame(convergence, columns=["Simulations", "DiscountedPrice"])
    os.makedirs(outdir, exist_ok=True)
    csv_path = os.path.join(outdir, f"{ticker}_convergence.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nConvergence data saved to {csv_path}")

    # plot convergence
    plt.figure(figsize=(9, 5))
    plt.plot(df["Simulations"], df["DiscountedPrice"], marker=".", linestyle="-", markersize=4, label=f"{ticker} Asian Call")
    # Add a horizontal line for the final price
    plt.axhline(price, color='r', linestyle='--', linewidth=1, label=f'Final Price: ${price:.4f}')
    plt.xlabel("Simulations")
    plt.ylabel("Discounted Option Price ($)")
    plt.title(f"Monte Carlo Convergence ({ticker})")
    plt.grid(True)
    plt.legend()
    png_path = os.path.join(outdir, f"{ticker}_convergence_plot.png")
    plt.tight_layout()
    plt.savefig(png_path, dpi=300)
    # plt.show() # Disabled plt.show() to prevent blocking in some environments
    plt.close()
    print(f"Convergence plot saved to {png_path}")

# --- Comparison Mode ---
def run_compare(tickers, T=1.0, r=0.01, steps=100, simulations=20000, seed=None, outdir="."):
    print(f"Running COMPARE mode for {len(tickers)} tickers (steps={steps}, sims={simulations})")
    results = []

    for ticker in tickers:
        try:
            S0, K, sigma = fetch_stock_params(ticker)
        except Exception as e:
            print(f"⚠️ Skipping {ticker} due to data error: {e}")
            continue

        price, ci, convergence = monte_carlo_asian_call(S0, K, T, sigma, r, steps, simulations, seed=seed)
        results.append({
            "ticker": ticker,
            "S0": S0,
            "K": K,
            "price": price,
            "sigma": sigma,
            "ci": ci,
            "convergence": convergence
        })

        # save each ticker's convergence to a CSV
        os.makedirs(outdir, exist_ok=True)
        df = pd.DataFrame(convergence, columns=["Simulations", "DiscountedPrice"])
        csv_path = os.path.join(outdir, f"{ticker}_convergence.csv")
        df.to_csv(csv_path, index=False)

    if not results:
        print("No valid results to show.")
        return

    # Print table
    headers = ["Ticker", "S0", "K", "Price", "Volatility", "95% CI"]
    table = []
    for r in results:
        table.append([
            r["ticker"],
            f"{r['S0']:.2f}",
            f"{r['K']:.2f}",
            f"{r['price']:.4f}",
            f"{r['sigma']:.4f}",
            f"[{r['ci'][0]:.4f}, {r['ci'][1]:.4f}]"
        ])
    print("\n" + "=" * 60)
    print("FINAL QUANTITATIVE COMPARISON")
    print("=" * 60)
    print(tabulate(table, headers=headers, tablefmt="github"))

    # bar chart for prices
    tick_labels = [r["ticker"] for r in results]
    prices = [r["price"] for r in results]

    plt.figure(figsize=(10, 6))
    plt.bar(tick_labels, prices)
    plt.ylabel("Discounted Option Price ($)")
    plt.title(f"Comparison of Asian Option Prices (T={T}y, r={r})")
    plt.tight_layout()
    png_path = os.path.join(outdir, f"comparison_plot.png")
    plt.savefig(png_path, dpi=300)
    # plt.show() # Disabled plt.show()
    plt.close()
    print(f"\nComparison plot saved to {png_path}")

# --- CLI Entrypoint ---
def parse_args():
    parser = argparse.ArgumentParser(description="Monte Carlo Asian Option Pricer")
    parser.add_argument("mode", choices=["single", "compare"], help="Run in single or compare mode")
    # Added a more explicit positional argument for tickers to handle single mode better
    parser.add_argument("tickers", nargs="*", help="Tickers for compare mode or optional ticker for single mode")
    parser.add_argument("--simulations", type=int, help="Number of Monte Carlo simulations")
    parser.add_argument("--steps", type=int, help="Number of time steps per path")
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    parser.add_argument("--outdir", type=str, default=".", help="Directory to save CSVs and plots")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    # defaults
    if args.mode == "single":
        ticker = args.tickers[0] if args.tickers else "SPY"
        sims = args.simulations if args.simulations is not None else 50000
        steps = args.steps if args.steps is not None else 100
        # T and r are currently hardcoded inside run_single (0.75 and 0.01)
        run_single(ticker=ticker, steps=steps, simulations=sims, seed=args.seed, outdir=args.outdir)
    else:  # compare
        if len(args.tickers) < 2:
            print("\nError: Please provide at least 2 tickers for comparison mode.")
            print("Example: python asian_montecarlo.py compare AAPL MSFT")
        else:
            sims = args.simulations if args.simulations is not None else 20000
            steps = args.steps if args.steps is not None else 100
            # T and r are currently hardcoded inside run_compare (1.0 and 0.01)
            run_compare(args.tickers, steps=steps, simulations=sims, seed=args.seed, outdir=args.outdir)