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


def _write_source(text):
    with open(os.path.join(DATA, "market_source.txt"), "w") as f:
        f.write(text)


def _load_or_generate():
    """Market-data priority:
      1. data/market_comp.csv      -- a local working file (e.g. REAL data freshly
                                       written by data/ingest_h1b.py).
      2. data/market_h1b.csv       -- the committed, de-identified REAL H-1B band
                                       sample that ships in the repo (so a fresh
                                       deploy like Replit trains on real bands).
      3. synthetic                 -- generated if neither exists.
    Offers are ALWAYS synthetic: accept/decline outcomes are internal data with no
    public ground truth, so the acceptance model is trained on a simulated process."""
    mpath = os.path.join(DATA, "market_comp.csv")
    h1bpath = os.path.join(DATA, "market_h1b.csv")
    opath = os.path.join(DATA, "offers.csv")

    if os.path.exists(mpath):
        market = pd.read_csv(mpath)
        # Preserve an existing provenance marker (ingest writes one); if absent,
        # this is likely the committed real sample copied into place.
        if _market_source() == "unknown":
            _write_source("REAL: DOL H-1B LCA (cleaned working file, base wage only)")
    elif os.path.exists(h1bpath):
        market = pd.read_csv(h1bpath)
        _write_source(f"REAL: DOL H-1B LCA committed sample ({len(market):,} "
                      f"de-identified records; offered BASE wage only, excl. equity/bonus)")
    else:
        print("Market data not found -- generating synthetic sample...")
        market, _ = generate()
        market.to_csv(mpath, index=False)
        _write_source("synthetic (grounded sample from data/generate_sample.py)")

    if os.path.exists(opath):
        offers = pd.read_csv(opath)
    else:
        print("Offers not found -- generating synthetic offers (no public source)...")
        _, offers = generate()
        offers.to_csv(opath, index=False)

    return market, offers


def ensure_trained(artifact_dir=ARTIFACTS):
    """Idempotent bootstrap for deployment (e.g. Replit): if model artifacts are
    missing, generate sample data as needed and train + save them. Returns the
    loaded models. On a repo with no committed data/artifacts, this makes the app
    work on first launch without a manual `python train.py` step."""
    from src.model import MarketModel
    from src.acceptance import AcceptanceModel
    if os.path.exists(os.path.join(artifact_dir, "market_model.joblib")):
        return MarketModel.load(artifact_dir), AcceptanceModel.load(artifact_dir)
    market, offers = _load_or_generate()
    market_model = MarketModel().fit(market)
    accept_model = AcceptanceModel().fit(offers)
    market_model.save(artifact_dir)
    accept_model.save(artifact_dir)
    return market_model, accept_model


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
