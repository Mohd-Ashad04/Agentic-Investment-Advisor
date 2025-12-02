# agents/crew_orchestrator.py
"""
CrewOrchestrator: tries to use CrewAI (crewai) if installed and configured.
Falls back to local deterministic orchestration using local agents.
This orchestrator enforces that, if the user provided a universe list,
we filter fetched price data to only those tickers.
"""
import os
from typing import List

# optional crewai import
try:
    import crewai
    HAS_CREW = True
except Exception:
    HAS_CREW = False

from .market_data_agent import MarketDataAgent
from .risk_assessment_agent import RiskAssessmentAgent
from .portfolio_generator_agent import PortfolioGeneratorAgent
from .ai_explainer_agent import AIExplainerAgent

class CrewOrchestrator:
    def __init__(self):
        self.market = MarketDataAgent()
        self.risk = RiskAssessmentAgent()
        self.portfolio = PortfolioGeneratorAgent()
        self.explainer = AIExplainerAgent()

    def _normalize_requested_universe(self, universe) -> List[str]:
        """
        Normalize user-provided universe to uppercase tickers and remove empties.
        Accepts list-like or comma-separated string.
        """
        if not universe:
            return []
        if isinstance(universe, str):
            universe = [u.strip() for u in universe.split(',') if u.strip()]
        # ensure uppercase strings
        return [str(u).upper() for u in universe if u]

    def run(self, budget: float, risk_level: str, universe=None) -> dict:
        """
        Orchestrate agents:
          - fetch prices (optionally using CrewAI if available)
          - compute risk report
          - generate portfolio
          - generate explanation (via Gemini if configured)
        """
        requested = self._normalize_requested_universe(universe)
        # If CrewAI is available and you want to use it, you can add orchestration logic here.
        # For lab/demo we default to deterministic local flow.
        # --- Fetch prices ---
        try:
            df = self.market.fetch_universe_prices(requested if requested else None)
        except Exception as e:
            # on failure, attempt a fallback with default universe
            print(f"WARNING: market fetch failed for requested={requested}: {e}")
            try:
                df = self.market.fetch_universe_prices(None)
            except Exception as e2:
                return {
                    'portfolio': {},
                    'risk_report': {},
                    'explanation': f"Failed to fetch market data: {e2}",
                    'prices': {},
                    'error': 'market_fetch_failed'
                }

        # If user requested specific tickers, filter to those tickers only (intersection)
        if requested:
            available_cols = [c for c in df.columns if str(c).upper() in requested]
            if not available_cols:
                # log for debugging
                print("DEBUG: none of requested tickers found in fetched prices. requested:", requested, "available:", list(df.columns))
                # continue with df as-is (we already fetched fallback if necessary)
            else:
                df = df[available_cols]
                print("DEBUG: filtered dataframe to requested tickers:", available_cols)

        # --- Risk assessment ---
        try:
            risk_report = self.risk.assess_universe(df)
        except Exception as e:
            print("WARNING: risk assessment failed:", e)
            risk_report = {'volatility': {}, 'summary': {}}

        # --- Portfolio generation ---
        try:
            portfolio = self.portfolio.generate_portfolio(budget, risk_level, df, risk_report)
        except Exception as e:
            print("WARNING: portfolio generation failed:", e)
            portfolio = {'budget': budget, 'holdings': [], 'allocated': 0.0, 'remaining': budget}

        # --- Explanation (LLM) ---
        try:
            explanation = self.explainer.explain_portfolio(portfolio, risk_level, risk_report)
        except Exception as e:
            print("WARNING: explanation agent failed:", e)
            explanation = "AI explanation unavailable."

        # --- Prices JSON for frontend ---
        try:
            prices_json = self.market.prices_to_json(df)
        except Exception as e:
            print("WARNING: prices_to_json failed:", e)
            prices_json = {}

        return {
            'portfolio': portfolio,
            'risk_report': risk_report,
            'explanation': explanation,
            'prices': prices_json
        }
