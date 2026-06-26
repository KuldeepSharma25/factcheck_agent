import os
import time

import pandas as pd
import streamlit as st
from anthropic import Anthropic

from factcheck_core import extract_claims, extract_text_from_pdf, verify_claim

st.set_page_config(page_title="Fact-Check Agent", page_icon="✅", layout="wide")

VERDICT_COLORS = {
    "Verified": "#1a7f37",
    "Inaccurate": "#b8860b",
    "False": "#c0392b",
}
VERDICT_ICONS = {
    "Verified": "✅",
    "Inaccurate": "⚠️",
    "False": "❌",
}


def get_api_key() -> str:
    """Pull the Anthropic API key from Streamlit secrets, env var, or user input."""
    if "ANTHROPIC_API_KEY" in st.secrets:
        return st.secrets["ANTHROPIC_API_KEY"]
    env_key = os.environ.get("ANTHROPIC_API_KEY")
    if env_key:
        return env_key
    return st.session_state.get("manual_api_key", "")


def render_header():
    st.title("✅ Fact-Check Agent")
    st.caption(
        "Upload a PDF. The agent extracts factual claims, checks each one "
        "against live web data, and flags anything outdated, wrong, or fabricated."
    )


def render_sidebar_key_input():
    api_key = get_api_key()
    if api_key:
        return api_key

    st.warning("No Anthropic API key found in secrets. Enter one below to run this app.")
    manual_key = st.text_input("Anthropic API key", type="password", key="manual_api_key_input")
    if manual_key:
        st.session_state["manual_api_key"] = manual_key
        return manual_key
    return ""


def render_verdict_badge(verdict: str) -> str:
    color = VERDICT_COLORS.get(verdict, "#666666")
    icon = VERDICT_ICONS.get(verdict, "❓")
    return f"<span style='color:{color}; font-weight:600;'>{icon} {verdict}</span>"


def main():
    render_header()
    api_key = render_sidebar_key_input()

    uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])

    run_clicked = st.button("Run Fact-Check", type="primary", disabled=not (uploaded_file and api_key))

    if not api_key:
        st.info("Add your Anthropic API key above to get started.")
        return

    if not uploaded_file:
        st.info("Upload a PDF to begin.")
        return

    if not run_clicked:
        return

    client = Anthropic(api_key=api_key)

    # Step 1: extract text
    with st.spinner("Reading PDF..."):
        document_text = extract_text_from_pdf(uploaded_file)

    if not document_text.strip():
        st.error("Could not extract any text from this PDF. It may be a scanned image without OCR text.")
        return

    # Step 2: extract claims
    with st.spinner("Extracting factual claims..."):
        claims = extract_claims(client, document_text)

    if not claims:
        st.warning("No specific, checkable factual claims were found in this document.")
        return

    st.success(f"Found {len(claims)} checkable claim(s). Verifying against live web data...")

    # Step 3: verify each claim, streaming results into the UI as they complete
    results = []
    progress_bar = st.progress(0)
    results_container = st.container()

    for i, claim_obj in enumerate(claims):
        claim_text = claim_obj.get("claim", "")
        with st.spinner(f"Verifying claim {i + 1} of {len(claims)}..."):
            verdict_obj = verify_claim(client, claim_text)

        row = {
            "claim": claim_text,
            "category": claim_obj.get("category", "other"),
            "source_snippet": claim_obj.get("source_snippet", ""),
            "verdict": verdict_obj.get("verdict", "False"),
            "correct_fact": verdict_obj.get("correct_fact", ""),
            "explanation": verdict_obj.get("explanation", ""),
            "sources": verdict_obj.get("sources", []),
        }
        results.append(row)
        progress_bar.progress((i + 1) / len(claims))

        with results_container:
            with st.expander(
                f"{VERDICT_ICONS.get(row['verdict'], '❓')}  {claim_text}",
                expanded=True,
            ):
                st.markdown(render_verdict_badge(row["verdict"]), unsafe_allow_html=True)
                st.markdown(f"**Original claim:** {row['claim']}")
                st.markdown(f"**Correct / current fact:** {row['correct_fact']}")
                st.markdown(f"**Why:** {row['explanation']}")
                if row["sources"]:
                    st.markdown("**Sources:** " + ", ".join(str(s) for s in row["sources"]))

    progress_bar.empty()

    # Summary table + download
    st.divider()
    st.subheader("Summary")
    df = pd.DataFrame(results)
    summary_counts = df["verdict"].value_counts()
    cols = st.columns(3)
    cols[0].metric("✅ Verified", int(summary_counts.get("Verified", 0)))
    cols[1].metric("⚠️ Inaccurate", int(summary_counts.get("Inaccurate", 0)))
    cols[2].metric("❌ False", int(summary_counts.get("False", 0)))

    st.dataframe(
        df[["claim", "verdict", "correct_fact", "category"]],
        use_container_width=True,
    )

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download full report as CSV",
        data=csv,
        file_name="fact_check_report.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
