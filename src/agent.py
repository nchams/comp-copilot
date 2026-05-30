"""The agent layer: turn model outputs into a written, reviewable offer
recommendation -- a first-pass draft a People leader edits, not a number they
have to derive from scratch.

Uses the Claude API when ANTHROPIC_API_KEY is set; otherwise falls back to a
deterministic template so the app always works (e.g. in a demo without a key).
"""

import os

def _ensure_env():
    """Load a local .env if present (maps 1:1 to Replit "Secrets" in production).
    Called on each check so a freshly-created .env is picked up without a server
    restart. Safe no-op if python-dotenv is missing or there's no .env file."""
    try:
        from dotenv import load_dotenv
        # override=True so editing .env takes effect on the next Streamlit rerun
        # without a server restart. On Replit there's no .env file, so the
        # Secrets-provided env vars are used unchanged.
        load_dotenv(override=True)
    except Exception:  # noqa: BLE001
        pass


_ensure_env()
MODEL = os.environ.get("COPILOT_MODEL", "claude-sonnet-4-6")


def claude_available() -> bool:
    """True if an API key is configured -- used by the UI to show live status."""
    _ensure_env()
    return bool(os.environ.get("ANTHROPIC_API_KEY"))

SYSTEM = """You are a compensation analyst assistant for a People/HR team. You \
draft FIRST-PASS offer recommendations that a human compensation partner will \
review and adjust. You never invent numbers -- you only use the figures given \
to you. Be concise, specific, and decision-oriented. Always present a primary \
recommendation plus one alternative structure (e.g. more equity, less base). \
Flag clearly when an offer sits below the market band. End with a one-line \
summary a recruiter can paste into an approval thread. Keep it under 220 words."""


def _facts(ctx: dict) -> str:
    b = ctx["band"]
    return (
        f"Role: {ctx['role_family']} | Level: {ctx['level_label']} | "
        f"Location: {ctx['metro']} | Candidate experience: {ctx['years_exp']} yrs\n"
        f"Proposed base offer: ${ctx['offer_base']:,.0f}\n"
        f"Market band (base): p25 ${b['p25']:,.0f} / p50 ${b['p50']:,.0f} / p75 ${b['p75']:,.0f}\n"
        f"Offer sits at the {ctx['percentile']:.0f}th percentile of market.\n"
        f"Predicted acceptance at this offer: {ctx['accept_prob']:.0%}\n"
        f"To reach ~{ctx['target']:.0%} acceptance: base of "
        f"${ctx['target_offer']:,.0f} (predicted accept {ctx['target_prob']:.0%}).\n"
    )


def draft_recommendation(ctx: dict) -> dict:
    """ctx must contain the keys referenced in _facts(). Returns
    {'text': str, 'source': 'claude'|'template'}."""
    _ensure_env()
    facts = _facts(ctx)
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=key)
            msg = client.messages.create(
                model=MODEL,
                max_tokens=600,
                system=SYSTEM,
                messages=[{
                    "role": "user",
                    "content": f"Draft an offer recommendation from these facts:\n\n{facts}",
                }],
            )
            return {"text": msg.content[0].text, "source": "claude"}
        except Exception as e:  # noqa: BLE001 -- degrade gracefully to template
            return {"text": _template(ctx) + f"\n\n_(LLM unavailable: {e})_",
                    "source": "template"}
    return {"text": _template(ctx), "source": "template"}


def _template(ctx: dict) -> str:
    b = ctx["band"]
    pos = ctx["percentile"]
    if pos < 35:
        verdict = f"**Below market** ({pos:.0f}th pct). Recommend raising base."
    elif pos > 70:
        verdict = f"**Strong** ({pos:.0f}th pct) -- competitive, room to hold."
    else:
        verdict = f"At-market ({pos:.0f}th pct)."
    return (
        f"### Draft recommendation\n"
        f"{verdict}\n\n"
        f"- **Proposed offer:** ${ctx['offer_base']:,.0f} base "
        f"(predicted acceptance {ctx['accept_prob']:.0%}).\n"
        f"- **Market band:** p25 ${b['p25']:,.0f} / p50 ${b['p50']:,.0f} / p75 ${b['p75']:,.0f}.\n"
        f"- **Primary rec:** move base to **${ctx['target_offer']:,.0f}** to reach "
        f"~{ctx['target_prob']:.0%} acceptance.\n"
        f"- **Alternative:** hold base near p50 (${b['p50']:,.0f}) and close the gap "
        f"with a sign-on / equity bump to preserve internal band equity.\n\n"
        f"_Summary: {ctx['role_family']} {ctx['level_label']} @ {ctx['metro']} — "
        f"offer at {pos:.0f}th pct; recommend base ${ctx['target_offer']:,.0f}._"
    )
