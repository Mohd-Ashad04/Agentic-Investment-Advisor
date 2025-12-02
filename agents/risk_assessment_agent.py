import numpy as np

class RiskAssessmentAgent:
    def assess_universe(self, prices_df):
        returns = prices_df.pct_change().dropna()
        vol = returns.std() * np.sqrt(252)
        drawdowns = {}
        for col in prices_df.columns:
            series = prices_df[col].dropna()
            if len(series)==0:
                drawdowns[col]=0.0
                continue
            roll_max = series.cummax()
            drawdown = ((series - roll_max)/roll_max).min()
            drawdowns[col]=float(drawdown)
        vol = vol.fillna(0)
        risk_score = (vol.rank(ascending=True)/len(vol))
        report = {
            'volatility': vol.to_dict(),
            'drawdown': drawdowns,
            'risk_score': risk_score.to_dict(),
            'summary': {
                'avg_volatility': float(vol.mean()),
                'max_volatility': float(vol.max()),
                'min_volatility': float(vol.min())
            }
        }
        return report
