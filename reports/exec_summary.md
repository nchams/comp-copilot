# Comp Co-Pilot — Executive Summary

**For:** People & Recruiting leadership · **From:** Data Science · **Status:** Prototype

## The problem
Offer decisions are made one spreadsheet at a time. A partner pulls a benchmark
table, eyeballs the band, and guesses what it takes to close. The result is slow,
inconsistent, and invisible at the org level — we can't see where we're
systematically below market (a retention risk) or above it (a budget leak).

## What we built
A system that, for any candidate profile, instantly returns the market band, the
predicted acceptance probability of a proposed offer, and the specific base
needed to hit a target close-rate — then has an AI agent draft the written
recommendation a partner reviews and edits.

## Three things it changes
1. **From lookup to model.** We estimate the full p25–p75 band for *any*
   role/level/metro/experience cell, including thinly-covered ones.
2. **From judgment to quantified trade-off.** "+\$9K base → acceptance 64% → 76%."
   Partners trade dollars against close-rate explicitly.
3. **From blank page to editable draft.** The agent writes the first pass
   (primary + an equity-weighted alternative). Humans stay the decision-maker.

## Does it work?
- Market-band model: p25–p75 **coverage ≈ 48%** on held-out data (target 50% — calibrated).
- Acceptance model: **AUC ≈ 0.75**.
- Causal check: being below market raises attrition **~12pts** *after* adjusting
  for confounders via propensity-score matching — the naive number (~15pts)
  overstates it. Below-market pay is a real, not spurious, retention lever.

## What it unlocks next
- Always-on agent: weekly Slack digest of roles drifting below market.
- Internal-equity guardrails: flag offers that compress a current employee's band.
- Plug into live offer + band systems (Ashby, Carta) behind access controls.

## A note on data
Built on **public proxies** (H-1B LCA disclosures, BLS OEWS) and synthetic data
by design — real comp data is confidential. The architecture is production-ready
to point at internal sources behind proper controls.
