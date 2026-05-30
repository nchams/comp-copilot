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

### Running on REAL H-1B LCA data

The market-band model can be trained on **actual US DOL H-1B LCA disclosure data** — real offered wages by job title, SOC code, and worksite. This path is fully built (`data/ingest_h1b.py`):

1. Download the latest **LCA Programs (H-1B, H-1B1, E-3)** disclosure `.xlsx` from the [DOL OFLC Performance Data page](https://www.dol.gov/agencies/eta/foreign-labor/performance) **in your browser** (DOL's CDN blocks scripted downloads), and save it into `data/raw/`.
2. `python data/ingest_h1b.py` — streams the file (handles the ~180MB annual file without loading it all into memory), keeps certified tech roles, normalizes wages to annual, levels free-text titles via `src/leveling.py`, buckets worksites into metros, and writes `data/market_comp.csv`.
3. `python train.py` — retrains the band model on the real data and records the provenance.

> **⚠️ Data-provenance caveat (important, and a talking point):** the LCA wage is the **offered base wage** on the petition. It *excludes equity and bonus* and clusters near prevailing-wage floors, so these real bands read **lower** than total-comp sources like Levels.fyi. The model is honest about what it measures: *base-wage* competitiveness, not total comp. The acceptance model stays synthetic — accept/decline outcomes are internal data with no public ground truth.

Real-data coverage is strongest for engineering/data roles (clean SOC codes: Software Developers `15-1252`, Data Scientists `15-2051`, etc.); Product Manager / Designer have no clean H-1B SOC mapping and are sparse on this source.

#### Real-data results (ran on an actual DOL LCA file)

Ingested an actual FY2026 LCA disclosure file: **scanned 1,039,355 records → kept 103,655** certified tech roles (Software Engineer 72.6K, Data Scientist 13.7K, Data Engineer 8.0K, Engineering Manager 6.0K). Retrained band model: **p25–p75 coverage 49.8%** (target 50% — well calibrated on real data).

Real **base-wage** bands, SF Bay Area (offered base only — *excludes equity/bonus*):

| Role | Level | p25 | p50 | p75 |
|---|---|---:|---:|---:|
| Software Engineer | L3 | $167K | $190K | $220K |
| Software Engineer | L5 | $175K | $203K | $229K |
| Data Scientist | L4 | $170K | $190K | $222K |
| Engineering Manager | L5 | $219K | $254K | $281K |

The headline finding is a **data-provenance lesson**: a Data Scientist L4 in SF shows a real H-1B base-wage p50 of ~$190K vs. ~$261K from total-comp-anchored synthetic data. That ~$70K gap is precisely the equity + bonus that LCA base-wage filings omit — so this model benchmarks *base-wage competitiveness*, and total-comp benchmarking would require layering an equity/bonus source on top.

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
data/ingest_h1b.py        REAL H-1B LCA xlsx -> cleaned market_comp.csv
data/download_data.py     real H-1B LCA + BLS OEWS downloader (URLs in sources.json)
analysis/causal_pay_attrition.py   PSM: does below-market pay cause attrition?
train.py             train + evaluate + save artifacts
app.py               Streamlit reference app
REPLIT_AGENT_PROMPT.md    spec to rebuild/deploy the UI on Replit
```

## Building this on Replit

> _Fill this in after you rebuild the UI with Replit Agent and deploy. This
> short reflection is the part that demonstrates the JD's "Experience building
> on Replit" bonus and "Interest in the future of AI-native organizations."_

**Live app:** `<paste your *.replit.app URL here>`

**Approach.** I did the data engineering and modeling locally (reproducible,
testable), then used **Replit Agent** to build and deploy the front-end around
the trained model artifacts — the same division of labor a People DS team would
use to ship an internal tool fast.

**What Replit Agent got right on the first pass:**
- `<e.g. scaffolded the full Streamlit layout + sidebar inputs from the spec>`
- `<e.g. wired the plotly band chart and acceptance curve correctly>`
- `<...>`

**Where I stepped in:**
- `<e.g. fixed quantile-crossing so p25 <= p50 <= p75 always holds>`
- `<e.g. tightened the agent prompt so it never invents numbers>`
- `<...>`

**Time from model artifact to deployed internal tool:** `<e.g. ~45 minutes>`

**What this implies for AI-native People ops.** `<1-2 sentences: e.g. a People
team member who isn't an engineer could ship their own decision-support tool
this way, collapsing the gap between "we need a model for X" and "it's live" from
weeks to an afternoon — which is exactly the operating model this role is meant
to build.>`

## Roadmap (production version)

- Point at live Ashby offer data + Carta band structure behind access controls.
- Add internal-equity checks (does this offer compress a current employee's band?).
- Always-on agent: weekly digest of roles trending below market → Slack to comp leads.
- Confidence intervals on bands from disclosure-data sample size per cell.
