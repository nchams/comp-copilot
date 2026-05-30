# 💰 Comp Co-Pilot

**A market-benchmarking + offer-recommendation system for People teams — built to be reviewed by humans, not to be a dashboard.**

A recruiter or compensation partner enters a candidate (role, level, location, experience) and a proposed offer. Comp Co-Pilot:

1. Estimates the **market band** (p25 / p50 / p75 base) for that profile from a model trained on public compensation data.
2. Shows **where the offer sits** in the band and its **predicted acceptance probability**.
3. Recommends the **specific base** needed to hit a target acceptance rate.
4. Has an **AI agent draft a written recommendation** (primary + alternative structure) that the human edits before sending for approval.

> This is a portfolio project built to mirror a real role: **Data Scientist, People @ Replit**. It deliberately implements the role's headline charter — *"Connect offer data, band position, acceptance rates, and market benchmarks into a live system that recommends specific adjustments"* and *"AI agents that draft first-pass recommendations… People leaders review and adjust rather than starting from scratch."*

---

## Executive summary (the 1-pager)

Compensation decisions today are slow and inconsistent: a partner pulls a benchmark spreadsheet, eyeballs a band, and guesses at what it takes to close. Comp Co-Pilot turns that into an **always-on system**:

- **Benchmarking is modeled, not looked up.** A quantile model predicts the full p25–p75 band for *any* role/level/metro/experience combination — including profiles with thin survey coverage — instead of relying on a static table.
- **Recommendations are quantified.** An acceptance model converts "this offer is below market" into "raising base by \$9K moves acceptance from 64% → 76%," so partners trade off dollars against close-rate explicitly.
- **The first draft is written for you.** An LLM agent produces a reviewable recommendation with a primary option and an alternative (e.g. hold base at p50, close the gap with equity to protect internal band equity). The human stays the decision-maker.
- **It compounds into org-level insight.** The same band model surfaces where the org is systematically below market (a retention risk) or above (a budget leak) — the input to comp-cycle and workforce-planning conversations.

**Validated on held-out data:** market-band model p25–p75 coverage ≈ 48% (target 50%, i.e. well-calibrated); acceptance model AUC ≈ 0.75. A causal-inference companion analysis (below) shows that being *below market* raises attrition by ~12pts **after** controlling for confounders — the naive estimate overstates it.

---

## Quickstart

```bash
pip install -r requirements.txt
python train.py            # generates sample data + trains & saves models
streamlit run app.py       # launch the app at http://localhost:8501
```

Optional — enable Claude-drafted recommendations (otherwise a template is used):

```bash
export ANTHROPIC_API_KEY=sk-ant-...      # PowerShell: $env:ANTHROPIC_API_KEY="..."
```

Run the causal analysis:

```bash
python analysis/causal_pay_attrition.py
```

---

## How it maps to the role

| Job responsibility | Where it lives in this repo |
|---|---|
| Connect offer data, band position, acceptance rates & benchmarks into a live recommendation system | `src/model.py` (bands) + `src/acceptance.py` (close-rate) + `app.py` |
| AI agents that draft first-pass comp recommendations for human review | `src/agent.py` (Claude API, with template fallback) |
| Strong SQL & Python, large operational datasets | `data/download_data.py` (real H-1B LCA + BLS OEWS) + `src/leveling.py` |
| Statistical foundation incl. **causal inference** | `analysis/causal_pay_attrition.py` (propensity-score matching) |
| LLMs in analytics workflows | `src/agent.py` |
| Communicate to executives | this exec summary + `reports/exec_summary.md` |
| Handle sensitive comp data with discretion | uses **public proxies** by design; see note below |
| Experience building on Replit | deploys on Replit; see `REPLIT_AGENT_PROMPT.md` |

---

## Data

The pipeline runs on a **synthetic-but-grounded** dataset by default (`data/generate_sample.py`) so it works instantly anywhere, including on Replit, without shipping large files. The numbers are anchored to publicly reported tech comp (Levels.fyi / BLS ranges).

To run on **real public data**, edit `data/sources.json` with current release URLs and run `python data/download_data.py`:

- **H-1B LCA Disclosure Data** (US DOL OFLC) — actual offered wages by employer, title, SOC code, worksite. The closest public proxy to real offer data.
- **BLS OEWS** — wage percentiles by occupation and metro, to ground "market percentile."

Free-text titles are normalized onto the leveling ladder by `src/leveling.py` (auditable keyword rules — no black box, which matters for high-stakes comp).

> **On sensitive data:** real compensation and attrition data is confidential. This project intentionally uses *public proxies* and synthetic data. The architecture is identical to a production system pointed at internal sources (Ashby, Rippling, Carta) behind proper access controls.

---

## Architecture

```
candidate + offer
      │
      ▼
 ┌───────────────┐   band (p25/p50/p75)   ┌────────────────────┐
 │  MarketModel  │ ─────────────────────▶ │                    │
 │ (quantile GBM)│                         │   Agent (Claude)   │ ─▶ editable
 └───────────────┘                         │  draft_recommend.  │    draft
 ┌───────────────┐   P(accept) + target    │                    │
 │ AcceptanceMdl │ ─────────────────────▶ │                    │
 │  (logistic)   │                         └────────────────────┘
 └───────────────┘
```

## Repo layout

```
src/config.py        role/level/metro taxonomy (single source of truth)
src/leveling.py      free-text title -> (role, level)
src/model.py         quantile market-band model
src/acceptance.py    offer-acceptance model + "offer for target accept" solver
src/agent.py         Claude-drafted recommendation (template fallback)
data/generate_sample.py   grounded synthetic data
data/download_data.py     real H-1B LCA + BLS OEWS downloader
analysis/causal_pay_attrition.py   PSM: does below-market pay cause attrition?
train.py             train + evaluate + save artifacts
app.py               Streamlit reference app
REPLIT_AGENT_PROMPT.md    spec to rebuild/deploy the UI on Replit
```

## Roadmap (production version)

- Point at live Ashby offer data + Carta band structure behind access controls.
- Add internal-equity checks (does this offer compress a current employee's band?).
- Always-on agent: weekly digest of roles trending below market → Slack to comp leads.
- Confidence intervals on bands from disclosure-data sample size per cell.
