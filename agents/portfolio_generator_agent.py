# agents/portfolio_generator_agent.py
import numpy as np
import pandas as pd

class PortfolioGeneratorAgent:
    """
    Generates a simple allocation given budget and risk level.
    This implementation is defensive:
      - Only uses tickers present in prices_df.
      - Handles missing prices.
      - Returns number of shares (integer) and allocated amounts.
    """

    def __init__(self, top_n: int = 5):
        self.top_n = top_n

    def _sanitize_tickers(self, prices_df: pd.DataFrame, vol_series: pd.Series):
        """
        Return intersection of tickers present in both the prices DataFrame and the volatility series.
        Preserves order of vol_series (rankable).
        """
        prices_cols = [str(c) for c in prices_df.columns]
        vol_index = [str(i) for i in vol_series.index]
        available = [t for t in vol_index if t in prices_cols]
        return available

    def generate_portfolio(self, budget: float, risk_level: str, prices_df: pd.DataFrame, risk_report: dict):
        """
        budget: total USD budget (float)
        risk_level: 'low' | 'moderate' | 'high'
        prices_df: DataFrame of prices (dates x tickers)
        risk_report: dict returned by RiskAssessmentAgent with key 'volatility' mapping ticker->vol
        """
        # Defensive: ensure prices_df has at least one column
        if prices_df is None or prices_df.empty:
            return {
                'budget': budget,
                'allocated': 0.0,
                'remaining': round(budget, 2),
                'holdings': [],
                'error': 'No price data available'
            }

        # Build volatility series safely
        vol = pd.Series(risk_report.get('volatility', {}))
        # If vol is empty, fallback to simple std on returns from prices_df
        if vol.empty:
            try:
                returns = prices_df.pct_change().dropna()
                vol = returns.std() * np.sqrt(252)
            except Exception:
                # last resort: equal-vols
                vol = pd.Series({c: 0.0 for c in prices_df.columns})

        # Restrict to tickers present in prices_df
        available = self._sanitize_tickers(prices_df, vol)
        if not available:
            # if nothing intersects, use the prices_df columns
            available = [str(c) for c in prices_df.columns]
            vol = vol.reindex(available).fillna(0.0)
        else:
            vol = vol.reindex(available).fillna(0.0)

        # Rank tickers by volatility (ascending => low vol first)
        ranked = vol.sort_values(ascending=True)

        n = min(self.top_n, len(ranked))
        if n == 0:
            return {
                'budget': budget,
                'allocated': 0.0,
                'remaining': round(budget, 2),
                'holdings': [],
                'error': 'No tickers available after filtering'
            }

        # Choose tickers according to risk_level
        if risk_level == 'low':
            chosen = list(ranked.index[:n])
            # Put big weight on the lowest volatility
            if n == 1:
                weights = np.array([1.0])
            else:
                weights = np.array([0.8] + [0.2/(n-1)]*(n-1))
        elif risk_level == 'high':
            chosen = list(ranked.index[-n:])
            # more equal but favor higher vol
            # create weights proportional to volatility (higher vol -> higher weight)
            vols = ranked.loc[chosen].values
            # avoid zero division
            vols = np.where(vols <= 0, 1e-6, vols)
            # invert to prefer higher vol: use vols / sum(vols)
            weights = vols / vols.sum()
            # normalize
            if weights.sum() == 0:
                weights = np.ones(len(chosen)) / len(chosen)
        else:  # moderate
            # pick a mix from low, median, high
            low_count = n // 2
            high_count = n - low_count
            low = list(ranked.index[:low_count])
            high = list(ranked.index[-high_count:]) if high_count > 0 else []
            chosen = low + high
            if len(chosen) == 0:
                chosen = list(ranked.index[:n])
            weights = np.ones(len(chosen)) / len(chosen)

        # Normalize weights
        weights = np.array(weights, dtype=float)
        if weights.sum() == 0:
            weights = np.ones_like(weights) / len(weights)
        else:
            weights = weights / weights.sum()

        # Last available prices (most recent)
        last_prices = prices_df.iloc[-1].to_dict()

        allocation = []
        remaining = float(budget)
        for ticker, w in zip(chosen, weights):
            price = float(last_prices.get(ticker, np.nan)) if ticker in last_prices else float('nan')
            if np.isnan(price) or price <= 0:
                shares = 0
                allocated_amt = 0.0
            else:
                amount = budget * float(w)
                shares = int(amount // price)
                allocated_amt = round(shares * price, 2)
            allocation.append({
                'ticker': ticker,
                'weight': float(round(float(w), 6)),
                'price': float(price) if not np.isnan(price) else None,
                'shares': int(shares),
                'allocated': allocated_amt
            })
            remaining -= allocated_amt

        portfolio = {
            'budget': round(float(budget), 2),
            'allocated': round(float(budget - remaining), 2),
            'remaining': round(float(remaining), 2),
            'holdings': allocation
        }
        return portfolio
