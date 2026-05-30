"""Comp Co-Pilot -- reference Streamlit app.

Run locally:   streamlit run app.py
On Replit:     this file is the entrypoint (see .replit).

This is the "human-in-the-loop" surface: a recruiter/People partner enters a
candidate + proposed offer, sees where it sits in the market band and its
predicted acceptance, and gets an AI-drafted recommendation they can edit
before sending for approval.
"""

import os
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.config import ROLE_FAMILIES, LEVELS, METROS, LEVEL_YEARS
from src.model import MarketModel
from src.acceptance import AcceptanceModel
from src.agent import draft_recommendation, claude_available

ARTIFACTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")

st.set_page_config(page_title="Comp Co-Pilot", page_icon="💰", layout="wide")


@st.cache_resource
def load_models():
    # On a fresh deploy (e.g. Replit) there are no committed artifacts/data;
    # bootstrap them on first launch so the app just works. If real H-1B data
    # was ingested locally, those artifacts are used instead.
    if not os.path.exists(os.path.join(ARTIFACTS, "market_model.joblib")):
        from train import ensure_trained
        with st.spinner("First launch: training models on sample data…"):
            return ensure_trained(ARTIFACTS)
    return MarketModel.load(ARTIFACTS), AcceptanceModel.load(ARTIFACTS)


market_model, accept_model = load_models()

st.title("💰 Comp Co-Pilot")
st.caption("Market benchmarking + offer recommendations — a first-pass draft People leaders review, not a dashboard.")

from train import _market_source  # noqa: E402
_src = _market_source()
if _src.startswith("REAL"):
    st.caption(f"📊 **Data:** {_src}")
elif _src != "unknown":
    st.caption(f"📊 **Data:** {_src}")

if market_model is None:
    st.error("No trained models found. Run `python train.py` first to create artifacts/.")
    st.stop()

with st.sidebar:
    if claude_available():
        st.success("🤖 Claude is active — recommendations are AI-drafted.")
    else:
        st.warning("Template mode — set `ANTHROPIC_API_KEY` to enable Claude.")
    st.header("Candidate & offer")
    role = st.selectbox("Role family", list(ROLE_FAMILIES))
    level = st.selectbox("Level", list(LEVELS), format_func=lambda l: LEVELS[l]["label"], index=3)
    metro = st.selectbox("Location", list(METROS))
    years = st.slider("Years of experience", 0.0, 25.0, float(LEVEL_YEARS[level]), 0.5)
    target = st.slider("Target acceptance probability", 0.50, 0.95, 0.75, 0.05)

    band = market_model.predict_band(role, level, metro, years)
    default_offer = int(round(band["p50"], -3))
    offer = st.number_input("Proposed base offer ($)", min_value=40_000,
                            max_value=900_000, value=default_offer, step=1_000)

# --- compute ---
pct = market_model.percentile_of(offer, role, level, metro, years)
accept_prob = accept_model.predict_proba(offer, band["p50"], level, years)
tgt_offer, tgt_prob = accept_model.offer_for_target(band["p50"], level, years, target=target)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Market median (p50)", f"${band['p50']:,.0f}")
c2.metric("Offer vs market", f"{pct:.0f}th pct",
          delta=f"${offer - band['p50']:,.0f} vs p50")
c3.metric("Predicted acceptance", f"{accept_prob:.0%}")
c4.metric(f"Base for ~{target:.0%} accept", f"${tgt_offer:,.0f}",
          delta=f"{tgt_prob:.0%} accept")

# --- band visualization ---
left, right = st.columns([3, 2])
with left:
    st.subheader("Where the offer sits in the band")
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[band["p75"] - band["p25"]], y=["Base"], base=[band["p25"]],
        orientation="h", marker_color="rgba(99,110,250,0.25)",
        name="p25–p75 band", hovertemplate="p25 $%{base:,.0f}<extra></extra>"))
    for label, val, color in [("p25", band["p25"], "#888"), ("p50", band["p50"], "#2ca02c"),
                              ("p75", band["p75"], "#888")]:
        fig.add_vline(x=val, line_dash="dot", line_color=color,
                      annotation_text=f"{label} ${val:,.0f}", annotation_position="top")
    fig.add_vline(x=offer, line_color="#d62728", line_width=3,
                  annotation_text=f"Offer ${offer:,.0f}", annotation_position="bottom")
    if tgt_offer:
        fig.add_vline(x=tgt_offer, line_color="#ff7f0e", line_dash="dash",
                      annotation_text=f"Target ${tgt_offer:,.0f}", annotation_position="bottom")
    fig.update_layout(height=220, showlegend=False, margin=dict(l=10, r=10, t=40, b=10),
                      xaxis_title="Base salary ($)", yaxis=dict(showticklabels=False))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Acceptance vs base offer")
    xs = [band["p50"] * f for f in [0.85 + 0.01 * i for i in range(41)]]
    ys = [accept_model.predict_proba(x, band["p50"], level, years) for x in xs]
    curve = go.Figure(go.Scatter(x=xs, y=ys, mode="lines", line_color="#636efa"))
    curve.add_vline(x=offer, line_color="#d62728", annotation_text="current offer")
    curve.update_layout(height=240, margin=dict(l=10, r=10, t=10, b=10),
                        xaxis_title="Base salary ($)", yaxis_title="P(accept)",
                        yaxis_range=[0, 1])
    st.plotly_chart(curve, use_container_width=True)

with right:
    st.subheader("🤖 Agent-drafted recommendation")
    ctx = {
        "role_family": role, "level_label": LEVELS[level]["label"], "metro": metro,
        "years_exp": years, "offer_base": offer, "band": band, "percentile": pct,
        "accept_prob": accept_prob, "target": target,
        "target_offer": tgt_offer, "target_prob": tgt_prob,
    }
    if st.button("Generate recommendation", type="primary", use_container_width=True):
        with st.spinner("Drafting..."):
            rec = draft_recommendation(ctx)
        st.session_state["rec"] = rec
    rec = st.session_state.get("rec")
    if rec:
        badge = "Claude" if rec["source"] == "claude" else "template (no API key)"
        st.caption(f"Source: {badge} — edit before sending for approval.")
        st.text_area("Editable draft", rec["text"], height=340)
    else:
        st.info("Click **Generate recommendation** for an AI first-pass draft. "
                "Set `ANTHROPIC_API_KEY` to use Claude; otherwise a template is used.")

st.divider()
st.caption("Synthetic data grounded in public sources (H-1B LCA, BLS OEWS). "
           "Sensitive comp data is handled via public proxies by design.")
