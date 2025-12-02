# app.py
from flask import Flask, render_template, request, jsonify
from agents.crew_orchestrator import CrewOrchestrator
import os
import logging

# Basic logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev')

orchestrator = CrewOrchestrator()

@app.route('/')
def index():
    """Render the main index page with the form."""
    return render_template('index.html')

@app.route('/results')
def results_page():
    """Render the results page (reads data from sessionStorage on client)."""
    return render_template('results.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    """API endpoint that runs the orchestrator and returns JSON."""
    try:
        data = request.json or {}
        budget = float(data.get('budget', 10000))
        risk_level = data.get('risk_level', 'moderate')
        universe = data.get('universe', None)

        logging.info("Received recommend request: budget=%s risk=%s universe=%s", budget, risk_level, universe)
        result = orchestrator.run(budget=budget, risk_level=risk_level, universe=universe)
        return jsonify(result)
    except Exception as e:
        logging.exception("Error running recommendation")
        # Return JSON error for the client to handle
        return jsonify({"error": "internal_server_error", "message": str(e)}), 500

# Optional health check
@app.route('/healthz')
def healthz():
    return jsonify({"status": "ok"}), 200

if __name__ == '__main__':
    # When running locally, set host to 0.0.0.0 so accessible from other devices if needed.
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
