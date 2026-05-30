"""Market compensation model.

Fits gradient-boosted *quantile* regressors to estimate the market base-salary
band (p25 / p50 / p75) for any (role, level, metro, years) profile. Quantile
loss is the right tool here: comp benchmarking is fundamentally about *where in
the band* an offer sits, not a single point estimate.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

QUANTILES = (0.25, 0.50, 0.75)
CAT_FEATURES = ["role_family", "metro"]
NUM_FEATURES = ["level", "years_exp"]
FEATURES = CAT_FEATURES + NUM_FEATURES
ARTIFACT = "market_model.joblib"


def _make_pipeline(alpha: float) -> Pipeline:
    pre = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES)],
        remainder="passthrough",
    )
    gbr = GradientBoostingRegressor(
        loss="quantile", alpha=alpha, n_estimators=300,
        max_depth=3, learning_rate=0.05, subsample=0.9, random_state=42,
    )
    return Pipeline([("pre", pre), ("gbr", gbr)])


class MarketModel:
    """Holds one quantile regressor per target percentile."""

    def __init__(self):
        self.models = {}

    def fit(self, df: pd.DataFrame):
        X, y = df[FEATURES], df["base_salary"].values
        for q in QUANTILES:
            self.models[q] = _make_pipeline(q).fit(X, y)
        return self

    def predict_band(self, role_family, level, metro, years_exp):
        X = pd.DataFrame([{
            "role_family": role_family, "metro": metro,
            "level": int(level), "years_exp": float(years_exp),
        }])
        band = {f"p{int(q*100)}": float(self.models[q].predict(X)[0]) for q in QUANTILES}
        # Enforce monotonic ordering (quantile crossing can happen with GBMs).
        p25, p50, p75 = sorted([band["p25"], band["p50"], band["p75"]])
        return {"p25": p25, "p50": p50, "p75": p75}

    def percentile_of(self, offer_base, role_family, level, metro, years_exp):
        """Where does a given offer sit in the band? Linear interp across the
        three known quantiles, extrapolated gently at the tails."""
        b = self.predict_band(role_family, level, metro, years_exp)
        pts = [(b["p25"], 25), (b["p50"], 50), (b["p75"], 75)]
        if offer_base <= pts[0][0]:
            # Below p25: scale toward an implied p0.
            span = pts[1][0] - pts[0][0]
            return max(1.0, 25 - 25 * (pts[0][0] - offer_base) / max(span, 1))
        if offer_base >= pts[2][0]:
            span = pts[2][0] - pts[1][0]
            return min(99.0, 75 + 25 * (offer_base - pts[2][0]) / max(span, 1))
        for (x0, p0), (x1, p1) in zip(pts, pts[1:]):
            if x0 <= offer_base <= x1:
                return p0 + (p1 - p0) * (offer_base - x0) / max(x1 - x0, 1)
        return 50.0

    def save(self, artifact_dir):
        os.makedirs(artifact_dir, exist_ok=True)
        joblib.dump(self.models, os.path.join(artifact_dir, ARTIFACT))

    @classmethod
    def load(cls, artifact_dir):
        m = cls()
        m.models = joblib.load(os.path.join(artifact_dir, ARTIFACT))
        return m


def evaluate(model: MarketModel, df: pd.DataFrame) -> dict:
    """Pinball loss per quantile + empirical coverage of the p25-p75 band."""
    out = {}
    X = df[FEATURES]
    for q in QUANTILES:
        pred = model.models[q].predict(X)
        err = df["base_salary"].values - pred
        out[f"pinball_p{int(q*100)}"] = float(np.mean(np.maximum(q * err, (q - 1) * err)))
    p25 = model.models[0.25].predict(X)
    p75 = model.models[0.75].predict(X)
    inside = ((df["base_salary"].values >= p25) & (df["base_salary"].values <= p75)).mean()
    out["p25_p75_coverage"] = float(inside)  # ~0.50 if well-calibrated
    return out
