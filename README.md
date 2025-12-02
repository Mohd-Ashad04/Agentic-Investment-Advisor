# Agentic Investment Advisor - CrewAI + Gemini (A3)

This project demonstrates a multi-agent (Crew-like) architecture integrated into a Flask web app. It includes:

- Market Data Agent (fetches prices via yfinance)
- Risk Assessment Agent (computes volatility)
- Portfolio Generator Agent (creates allocations)
- CrewAI-style orchestration (uses `crewai` if available; otherwise falls back to deterministic implementation)
- Gemini (google-generativeai) used by an explanation agent

## How to run
1. Create virtualenv:
   - `python -m venv venv`
2. Activate:
   - Windows: `venv\\Scripts\\activate`
   - Mac/Linux: `source venv/bin/activate`
3. Install:
   - `pip install -r requirements.txt`
4. Copy `.env.example` to `.env` and set `GEMINI_API_KEY` (and CrewAI keys if you have them)
5. Run:
   - `python app.py`
6. Open: http://127.0.0.1:5000

## Notes for lab
- The `crew_orchestrator.py` shows how you would wire up CrewAI-like tasks. If you have real CrewAI credentials and library, set them and the app will attempt to use Crew.
- All AI calls have safe fallbacks for offline/demo use.
