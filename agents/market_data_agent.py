# agents/market_data_agent.py
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

class MarketDataAgent:
    """
    Robust market data agent that fetches historical prices and exposes a
    helper to convert a DataFrame to a JSON-friendly object for the frontend.
    """

    def __init__(self, lookback_days: int = 180):
        self.lookback_days = lookback_days

    def default_universe(self):
        return ['AAPL','MSFT','GOOGL','AMZN','TSLA','JNJ','V','PG','XOM','JPM']

    def _normalize_yf_output(self, raw: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize output from yfinance.download to a DataFrame of prices
        with tickers as columns and Date index.

        Handles cases:
        - Series (single ticker)
        - DataFrame with 'Adj Close' column
        - DataFrame with MultiIndex columns like (ticker, field)
        - Fallback to 'Close' if 'Adj Close' not present
        """
        if raw is None or raw.empty:
            raise ValueError("No data returned from yfinance. Check tickers and network.")

        # If single Series, turn into DataFrame
        if isinstance(raw, pd.Series):
            return raw.to_frame(name=str(raw.name) or "PRICE")

        # If 'Adj Close' exists at top-level
        if 'Adj Close' in raw.columns:
            df = raw['Adj Close']
            if isinstance(df, pd.Series):
                df = df.to_frame(name=df.name)
            return df

        # If MultiIndex columns (ticker, field)
        if isinstance(raw.columns, pd.MultiIndex):
            # level 1 fields might contain 'Adj Close' or 'Close'
            fields = list(raw.columns.levels[1])
            if 'Adj Close' in fields:
                df = raw.xs('Adj Close', axis=1, level=1)
                return df
            if 'Close' in fields:
                df = raw.xs('Close', axis=1, level=1)
                return df

        # If 'Close' exists at top-level
        if 'Close' in raw.columns:
            df = raw['Close']
            if isinstance(df, pd.Series):
                df = df.to_frame(name=df.name)
            return df

        # Last resort: extract numeric columns and return them (may be a wide numeric df)
        numeric = raw.select_dtypes(include='number')
        if not numeric.empty:
            return numeric

        # If still nothing, raise a helpful error
        raise KeyError(f"Couldn't find 'Adj Close' or 'Close' in yfinance output. Columns: {list(raw.columns)}")

    def fetch_universe_prices(self, universe=None) -> pd.DataFrame:
        """
        Fetch price history for the universe of tickers.
        Returns a DataFrame with tickers as columns and Date index.
        """
        if not universe:
            universe = self.default_universe()

        end = datetime.now()
        start = end - timedelta(days=self.lookback_days)

        # Explicitly set auto_adjust or prepost if you want; set progress=False for no output.
        # yfinance now defaults auto_adjust to True in future; we set it explicitly to avoid warnings.
        raw = yf.download(
            tickers=universe,
            start=start.strftime('%Y-%m-%d'),
            end=end.strftime('%Y-%m-%d'),
            progress=False,
            auto_adjust=False,  # keep raw fields (we'll select 'Adj Close' or 'Close')
            threads=True
        )

        # Normalize and return
        df = self._normalize_yf_output(raw)

        # Ensure DataFrame columns are strings (tickers)
        df.columns = [str(c) for c in df.columns]

        # Sort index ascending and drop all-NaN columns if any
        df = df.sort_index().dropna(axis=1, how='all')

        if df.empty:
            raise ValueError("No usable price columns after normalization. Check tickers/network.")

        return df

    def prices_to_json(self, df: pd.DataFrame, tail: int = 90) -> dict:
        """
        Convert prices DataFrame to a JSON-friendly dict for the frontend.
        - tail: number of recent days to include
        Returns: { 'dates': [...], 'TICKER': [price,...], ... }
        """
        # Defensive copy, forward/backfill to handle missing values
        df2 = df.copy().ffill().bfill()

        # If tail requested more than available rows, just use df2 as-is
        if tail is not None and len(df2) > tail:
            df2 = df2.tail(tail)

        # Round prices to 2 decimals for smaller payload
        out = {col: df2[col].round(2).tolist() for col in df2.columns}
        out['dates'] = [d.strftime('%Y-%m-%d') for d in df2.index]

        return out
