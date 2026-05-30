"""Generate a realistic synthetic compensation dataset so the whole pipeline
runs instantly anywhere -- including on Replit -- without shipping the large
(hundreds of MB) real H-1B / BLS files.

The generator is grounded in the taxonomy in src/config.py. It produces two CSVs:

  data/market_comp.csv   -- "market" observations (role, level, metro, yrs, base)
                            the percentile model is trained on.
  data/offers.csv        -- historical offers with an `accepted` label, used to
                            train the offer-acceptance model.

Swap this out for data/download_data.py + a real leveling pass when you want to
run on actual disclosure data; the downstream code is identical.
"""

import os
import numpy as np
import pandas as pd

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import ROLE_FAMILIES, LEVELS, METROS, LEVEL_YEARS, RANDOM_SEED

HERE = os.path.dirname(os.path.abspath(__file__))


def _market_base(role, level, metro, years, rng):
    """True market base salary for a profile, with lognormal noise."""
    anchor = ROLE_FAMILIES[role]["base_p50"]
    spread = ROLE_FAMILIES[role]["spread"]
    mult = LEVELS[level]["mult"] * METROS[metro]
    # Within-band experience effect: being above the typical years for the
    # level nudges pay up, capped so it doesn't dominate the level signal.
    typical = LEVEL_YEARS[level]
    exp_factor = 1.0 + np.clip((years - typical) / 25.0, -0.12, 0.18)
    mean = anchor * mult * exp_factor
    return float(rng.lognormal(mean=np.log(mean), sigma=spread * 0.45))


def generate(n_market: int = 9000, n_offers: int = 4000):
    rng = np.random.default_rng(RANDOM_SEED)
    roles = list(ROLE_FAMILIES)
    metros = list(METROS)
    levels = list(LEVELS)

    # ---- market observations ----
    rows = []
    for _ in range(n_market):
        role = rng.choice(roles, p=_role_weights(roles))
        level = int(rng.choice(levels, p=[0.10, 0.18, 0.30, 0.24, 0.12, 0.06]))
        metro = rng.choice(metros)
        years = max(0.0, rng.normal(LEVEL_YEARS[level], 1.8))
        base = _market_base(role, level, metro, years, rng)
        rows.append((role, level, metro, round(years, 1), round(base, -2)))
    market = pd.DataFrame(rows, columns=["role_family", "level", "metro", "years_exp", "base_salary"])

    # ---- offers with acceptance labels ----
    # Acceptance is driven by how the offer compares to the market median for
    # the profile (the gap), plus seniority (senior folks are more gap-sensitive)
    # and random competing-offer pressure. This is the *true* process the
    # acceptance model later has to recover.
    med_lookup = (
        market.groupby(["role_family", "level", "metro"])["base_salary"].median().to_dict()
    )
    orows = []
    for _ in range(n_offers):
        role = rng.choice(roles, p=_role_weights(roles))
        level = int(rng.choice(levels, p=[0.10, 0.18, 0.30, 0.24, 0.12, 0.06]))
        metro = rng.choice(metros)
        years = max(0.0, rng.normal(LEVEL_YEARS[level], 1.8))
        med = med_lookup.get((role, level, metro)) or _market_base(role, level, metro, years, rng)
        # Offers are made anywhere from ~12% below to ~12% above median.
        offer = med * rng.uniform(0.88, 1.12)
        gap = (offer - med) / med
        competing = rng.normal(0, 0.06)  # unobserved competing-offer pressure
        senior_sensitivity = 1.0 + 0.10 * (level - 3)
        logit = 1.6 + 9.0 * gap * senior_sensitivity - 12.0 * max(0, -gap) - 5.0 * competing
        p = 1 / (1 + np.exp(-logit))
        accepted = int(rng.random() < p)
        orows.append((role, level, metro, round(years, 1), round(offer, -2), round(med, -2), accepted))
    offers = pd.DataFrame(
        orows,
        columns=["role_family", "level", "metro", "years_exp", "offer_base", "market_p50_at_offer", "accepted"],
    )
    return market, offers


def _role_weights(roles):
    # Engineering-heavy org shape.
    w = {"Software Engineer": 0.40, "Data Engineer": 0.12, "Data Scientist": 0.12,
         "Product Manager": 0.14, "Product Designer": 0.12, "Engineering Manager": 0.10}
    arr = np.array([w[r] for r in roles])
    return arr / arr.sum()


if __name__ == "__main__":
    market, offers = generate()
    market.to_csv(os.path.join(HERE, "market_comp.csv"), index=False)
    offers.to_csv(os.path.join(HERE, "offers.csv"), index=False)
    with open(os.path.join(HERE, "market_source.txt"), "w") as f:
        f.write("synthetic (grounded sample from data/generate_sample.py)")
    print(f"Wrote {len(market):,} market rows -> data/market_comp.csv")
    print(f"Wrote {len(offers):,} offer rows  -> data/offers.csv "
          f"({offers['accepted'].mean():.1%} accepted)")
