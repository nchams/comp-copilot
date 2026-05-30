"""End-to-end training entrypoint.

    python train.py

Generates the sample data if missing, trains the market-percentile and
acceptance models, prints evaluation metrics, and writes artifacts to
artifacts/ for the app (and for Replit) to load.
"""

import os
import pandas as pd

from data.generate_sample import generate
from src.model import MarketModel, evaluate as eval_market
from src.acceptance import AcceptanceModel, evaluate as eval_accept

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data")
ARTIFACTS = os.path.join(ROOT, "artifacts")


def _market_source():
    p = os.path.join(DATA, "market_source.txt")
    return open(p).read().strip() if os.path.exists(p) else "unknown"


def _load_or_generate():
    """Market data is used as-is if present (it may be REAL H-1B data written by
    data/ingest_h1b.py); only generated if missing. Offers are ALWAYS synthetic:
    accept/decline outcomes are internal data with no public ground truth, so the
    acceptance model is trained on a simulated-but-realistic process."""
    mpath = os.path.join(DATA, "market_comp.csv")
    opath = os.path.join(DATA, "offers.csv")

    if os.path.exists(mpath):
        market = pd.read_csv(mpath)
    else:
        print("Market data not found -- generating synthetic sample...")
        market, _ = generate()
        market.to_csv(mpath, index=False)

    if os.path.exists(opath):
        offers = pd.read_csv(opath)
    else:
        print("Offers not found -- generating synthetic offers (no public source)...")
        _, offers = generate()
        offers.to_csv(opath, index=False)

    return market, offers


def main():
    market, offers = _load_or_generate()
    print(f"Market data source: {_market_source()}")
    print(f"Loaded {len(market):,} market rows, {len(offers):,} offers "
          f"({offers['accepted'].mean():.1%} accepted, synthetic).")

    # Hold out for honest metrics.
    m_train = market.sample(frac=0.8, random_state=1)
    m_test = market.drop(m_train.index)
    o_train = offers.sample(frac=0.8, random_state=1)
    o_test = offers.drop(o_train.index)

    print("\nTraining market percentile model...")
    market_model = MarketModel().fit(m_train)
    mm = eval_market(market_model, m_test)
    print(f"  pinball p25/p50/p75: {mm['pinball_p25']:.0f} / "
          f"{mm['pinball_p50']:.0f} / {mm['pinball_p75']:.0f}")
    print(f"  p25-p75 coverage: {mm['p25_p75_coverage']:.1%} (target ~50%)")

    print("\nTraining acceptance model...")
    accept_model = AcceptanceModel().fit(o_train)
    am = eval_accept(accept_model, o_test)
    print(f"  AUC: {am['auc']:.3f} | Brier: {am['brier']:.3f}")

    market_model.save(ARTIFACTS)
    accept_model.save(ARTIFACTS)
    print(f"\nArtifacts written to {ARTIFACTS}/")

    # Quick smoke-test of the full recommendation path.
    from src.config import LEVELS
    band = market_model.predict_band("Data Scientist", 4, "SF Bay Area", 7)
    offer = band["p50"] * 0.95
    pct = market_model.percentile_of(offer, "Data Scientist", 4, "SF Bay Area", 7)
    p = accept_model.predict_proba(offer, band["p50"], 4, 7)
    tgt_offer, tgt_p = accept_model.offer_for_target(band["p50"], 4, 7, target=0.75)
    print("\nSmoke test -- Data Scientist L4 (Senior), SF, 7 yrs:")
    print(f"  band p25/p50/p75: ${band['p25']:,.0f} / ${band['p50']:,.0f} / ${band['p75']:,.0f}")
    print(f"  offer ${offer:,.0f} -> {pct:.0f}th pct, accept {p:.0%}")
    print(f"  to hit 75% accept: base ${tgt_offer:,.0f} (accept {tgt_p:.0%})")


if __name__ == "__main__":
    main()
