"""Offer-acceptance model.

Learns P(accept | offer) as a function of how the offer compares to the market
median (the "gap"), seniority, and experience. This is what turns the benchmark
into an *actionable* recommendation: "raise base by $X and acceptance goes from
52% to 71%."
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ARTIFACT = "acceptance_model.joblib"


def _features(df: pd.DataFrame) -> pd.DataFrame:
    gap = (df["offer_base"] - df["market_p50_at_offer"]) / df["market_p50_at_offer"]
    return pd.DataFrame({
        "gap_pct": gap,
        "gap_pct_sq": gap ** 2,
        "level": df["level"],
        "years_exp": df["years_exp"],
        "gap_x_level": gap * df["level"],
    })


class AcceptanceModel:
    def __init__(self):
        self.pipe = Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000)),
        ])

    def fit(self, offers: pd.DataFrame):
        self.pipe.fit(_features(offers), offers["accepted"].values)
        return self

    def predict_proba(self, offer_base, market_p50, level, years_exp):
        df = pd.DataFrame([{
            "offer_base": offer_base, "market_p50_at_offer": market_p50,
            "level": int(level), "years_exp": float(years_exp),
        }])
        return float(self.pipe.predict_proba(_features(df))[0, 1])

    def offer_for_target(self, market_p50, level, years_exp, target=0.75,
                         lo=0.80, hi=1.25):
        """Smallest offer (as a fraction of market p50, within [lo, hi]) that
        reaches the target acceptance probability. Returns (offer_base, prob)."""
        grid = np.linspace(lo, hi, 91)
        best = None
        for frac in grid:
            offer = market_p50 * frac
            p = self.predict_proba(offer, market_p50, level, years_exp)
            if p >= target:
                return offer, p
            best = (offer, p)
        return best  # target unreachable within range -> return top of range

    def save(self, artifact_dir):
        os.makedirs(artifact_dir, exist_ok=True)
        joblib.dump(self.pipe, os.path.join(artifact_dir, ARTIFACT))

    @classmethod
    def load(cls, artifact_dir):
        m = cls()
        m.pipe = joblib.load(os.path.join(artifact_dir, ARTIFACT))
        return m


def evaluate(model: AcceptanceModel, offers: pd.DataFrame) -> dict:
    from sklearn.metrics import roc_auc_score, brier_score_loss
    p = model.pipe.predict_proba(_features(offers))[:, 1]
    y = offers["accepted"].values
    return {"auc": float(roc_auc_score(y, p)), "brier": float(brier_score_loss(y, p))}
