from agents.crew_orchestrator import CrewOrchestrator

def test_run():
    orch = CrewOrchestrator()
    res = orch.run(10000, 'moderate', ['AAPL','MSFT'])
    assert 'portfolio' in res
    assert 'prices' in res
