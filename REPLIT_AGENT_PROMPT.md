# Building the app half on Replit (your "built on Replit" story)

The local repo gives you a clean **data + model artifact** (`artifacts/*.joblib`) and a working reference app (`app.py`). The strongest version of this portfolio piece is to **rebuild and deploy the app layer on Replit using Replit Agent** — that's what earns the JD's "Experience building on Replit" bonus and demonstrates you understand Replit as a *user* of its flagship agentic product.

Two ways to do it; do whichever tells the better story for you.

## Option A — Import the repo, deploy as-is
1. Create a new Repl → "Import from GitHub" (push this repo first) **or** upload the folder.
2. Replit reads `.replit` and installs `requirements.txt`.
3. In the **Secrets** tab, add `ANTHROPIC_API_KEY` to enable Claude-drafted recommendations.
4. Open a shell, run `python train.py` once to create `artifacts/`.
5. Click **Run**, then **Deploy** (Autoscale / Cloud Run target is preconfigured in `.replit`).

## Option B — Rebuild the UI with Replit Agent (best narrative)
Keep the local `src/` model code, then paste the prompt below into **Replit Agent** to have it construct the app around your artifacts. Then iterate with the Agent and write up what it nailed vs. where you stepped in — that reflection is the part hiring managers love.

---

### Replit Agent prompt (paste this)

> Build a Streamlit app called **Comp Co-Pilot** for a People/HR compensation workflow. Assume there are two pre-trained model artifacts in `artifacts/`: `market_model.joblib` (a dict of scikit-learn quantile-regression pipelines keyed by 0.25/0.5/0.75 that predict base-salary percentiles) and `acceptance_model.joblib` (a scikit-learn pipeline predicting P(offer accepted)). Helper modules exist in `src/`: `model.MarketModel.load(dir)`, `acceptance.AcceptanceModel.load(dir)`, `agent.draft_recommendation(ctx)`, and `config` (which exposes `ROLE_FAMILIES`, `LEVELS`, `METROS`, `LEVEL_YEARS`).
>
> The app should:
> 1. In a sidebar, let the user pick role family, level, location (metro), years of experience, a target acceptance probability, and enter a proposed base offer.
> 2. Show four KPI metrics: market median (p50), offer-vs-market percentile, predicted acceptance probability, and the base needed to hit the target acceptance.
> 3. Visualize the p25–p75 band as a horizontal range with markers for p25/p50/p75, the current offer, and the target offer.
> 4. Plot an acceptance-vs-base curve with the current offer marked.
> 5. Have a "Generate recommendation" button that calls `agent.draft_recommendation(ctx)` and shows the returned text in an **editable** text area, labeled as an AI first-pass draft for human review. Read `ANTHROPIC_API_KEY` from Secrets; fall back to a template if absent.
> 6. Include a footer noting the data is synthetic, grounded in public sources (H-1B LCA, BLS OEWS), and that sensitive comp data is handled via public proxies by design.
>
> Use plotly for charts, cache model loading with `@st.cache_resource`, and configure it to run on port 8080 bound to 0.0.0.0 so it deploys on Replit. After building, add a Deploy config targeting Autoscale.

---

### What to write up afterward (the differentiator)
A short "Building this on Replit" section in your README or a LinkedIn post:
- What Replit Agent generated correctly on the first pass.
- Where you intervened (e.g. fixing the quantile-crossing guard, tuning the band chart).
- How fast you went from artifact → deployed internal tool.
- One sentence on what this implies for AI-native People ops — i.e. that non-engineers on a People team could ship their own decision tools this way.
