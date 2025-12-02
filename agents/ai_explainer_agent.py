import os

# Use google.generativeai if available
try:
    import google.generativeai as genai
    HAS_GENAI = True
except Exception:
    HAS_GENAI = False

class AIExplainerAgent:
    def __init__(self):
        self.key = os.getenv('GEMINI_API_KEY')
        if HAS_GENAI and self.key:
            genai.configure(api_key=self.key)
            # choose an appropriate model available to your account
            # in examples we will call genai directly when needed

    def explain_portfolio(self, portfolio, risk_level, risk_report=None):
        prompt = f"""You are a helpful financial advisor.
Given the risk level: {risk_level}
and the portfolio: {portfolio}
Provide a concise, clear explanation of why these tickers were chosen, the risk considerations, and any simple suggestions."""
        if HAS_GENAI and self.key:
            try:
                model = genai.get_model('models/text-bison-001')
                resp = model.generate(prompt=prompt)
                # some genai SDKs return a 'candidates' list with 'content'
                text = ''
                if hasattr(resp, 'candidates') and resp.candidates:
                    text = '\n'.join([c.content for c in resp.candidates])
                elif hasattr(resp, 'output'):
                    text = str(resp.output)
                else:
                    text = str(resp)
                return text
            except Exception as e:
                return f"AI unavailable: {e}"
        # fallback deterministic explanation
        tickers = [h['ticker'] for h in portfolio.get('holdings', [])]
        return f"Selected tickers: {', '.join(tickers)}. Portfolio matches {risk_level} risk preference. (No LLM available)"
