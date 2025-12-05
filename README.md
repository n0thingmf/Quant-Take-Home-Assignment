# Polymarket Quantitative Researcher – Take-Home Assignment

This folder contains my full solution to the Polymarket market-making assignment.

## Files

- `Take_Home_Assignment.ipynb`  
  Jupyter notebook with:
  - Part 1: core data analysis (spreads, volatility, flow/drift methodology).
  - Part 2: quoting logic, inventory control, refresh rules.
  - Part 3: simple market-making backtest on five markets, plus plots, summary + interpretation of simulation results.
  - Part 4: explanation of negative-risk markets and a simple rule-based detector of informed trading.

- `sim_mm.py`  
  Self-contained Python module implementing the toy Polymarket market maker used in the notebook:
  - `MMConfig`: configuration dataclass.
  - `SimplePolymarketMM`: quoting, price-path, and fill simulation.
  - `summarize_results`: helper to aggregate per-market statistics.

## How to run

1. Install Python 3.10+ and create a virtualenv.
2. Install standard scientific stack (at minimum: `pandas`, `numpy`, `matplotlib`).  
   The code does **not** depend on Polymarket APIs or `web3`; it only uses the provided CSVs.
3. Place the four provided CSVs from the assignment in the same directory:
   - `Polymarket data-interview - Hyperparameters.csv`
   - `Polymarket data-interview - Full Markets.csv`
   - `Polymarket data-interview - Volatility Markets.csv`
   - `Polymarket data-interview - All Markets.csv`
4. Open `polymarket_analysis_and_simulation.ipynb` in Jupyter / VSCode / etc and run all cells.

The notebook will:
- Reproduce the spread and volatility analysis.
- Run the market-making simulation across five markets and print the summary statistics.
- Plot PnL, inventory, and mid-price time series for inspection.