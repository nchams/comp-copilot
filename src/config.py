"""Shared taxonomy: role families, leveling ladder, and metro cost-of-labor factors.

These constants are the single source of truth for both the synthetic data
generator and the modeling / app layers, so the leveling framework stays
consistent end to end. The numbers are grounded in publicly reported tech
compensation (Levels.fyi, BLS OEWS, H-1B LCA disclosures) but are deliberately
approximate -- the point of the project is the *system*, not exact figures.
"""

# Role families we benchmark. base_p50 is an anchor: the market-median BASE
# salary for the IC3 level in a "tier-2" metro (multiplier 1.0). Everything else
# is derived from this anchor via level + metro + experience adjustments.
ROLE_FAMILIES = {
    "Software Engineer": {"base_p50": 165_000, "spread": 0.18},
    "Data Scientist": {"base_p50": 168_000, "spread": 0.19},
    "Product Manager": {"base_p50": 172_000, "spread": 0.20},
    "Product Designer": {"base_p50": 150_000, "spread": 0.18},
    "Data Engineer": {"base_p50": 160_000, "spread": 0.18},
    "Engineering Manager": {"base_p50": 205_000, "spread": 0.22},
}

# Leveling ladder. The multiplier is applied to the role's IC3 anchor.
# This is the "band" structure comp benchmarking revolves around.
LEVELS = {
    1: {"label": "L1 / Entry", "mult": 0.62},
    2: {"label": "L2 / Junior", "mult": 0.80},
    3: {"label": "L3 / Mid", "mult": 1.00},
    4: {"label": "L4 / Senior", "mult": 1.28},
    5: {"label": "L5 / Staff", "mult": 1.62},
    6: {"label": "L6 / Principal", "mult": 2.05},
}

# Metro cost-of-labor multipliers, anchored so tier-2 == 1.0.
METROS = {
    "SF Bay Area": 1.22,
    "New York City": 1.15,
    "Seattle": 1.12,
    "Boston": 1.08,
    "Austin": 1.00,
    "Denver": 0.97,
    "Remote (US)": 1.03,
    "Other US Metro": 0.94,
}

# Typical years of experience by level -- used to add a within-band signal.
LEVEL_YEARS = {1: 0.5, 2: 2, 3: 4, 4: 7, 5: 11, 6: 16}

RANDOM_SEED = 42
