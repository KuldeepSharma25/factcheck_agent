"""
Core logic for the Fact-Check Agent.

Pipeline:
1. extract_text_from_pdf  -> pulls raw text out of an uploaded PDF
2. extract_claims         -> asks Claude to identify discrete, checkable factual claims
3. verify_claim           -> asks Claude (with web search enabled) to check one claim
                             against live web data and return a verdict
"""

import json
import re
from typing import List, Dict, Any

import pdfplumber
from anthropic import Anthropic

MODEL_NAME = "claude-sonnet-4-6"


# ---------------------------------------------------------------------------
# 1. PDF TEXT EXTRACTION
# ---------------------------------------------------------------------------
def extract_text_from_pdf(file_obj) -> str:
    """Extract plain text from an uploaded PDF file-like object."""
    text_parts = []
    with pdfplumber.open(file_obj) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            text_parts.append(page_text)
    return "\n".join(text_parts)


# ---------------------------------------------------------------------------
# 2. CLAIM EXTRACTION
# ---------------------------------------------------------------------------
EXTRACTION_SYSTEM_PROMPT = """You are a precise fact-extraction engine.

You will be given the raw text of a marketing or business document. Your job
is to identify every discrete, independently checkable factual claim in it —
specifically statistics, percentages, dates, financial figures, market-size
numbers, technical specifications, or named comparisons ("X% faster than Y").

Rules:
- Only extract claims that are SPECIFIC and CHECKABLE against real-world,
  public information (a number, a date, a named fact). Skip vague marketing
  fluff ("we are the best", "industry leading") that has no checkable fact.
- Keep each claim as a short, self-contained sentence that includes enough
  context to verify it standalone (e.g. include the subject/company/product
  name, not just "grew by 40%").
- Return STRICT JSON only — an array of objects, no prose, no markdown fences.
- Each object must have exactly these fields:
  {
    "claim": "<the self-contained claim text>",
    "category": "<one of: statistic, date, financial, technical, other>",
    "source_snippet": "<the exact original sentence/phrase from the document>"
  }
- If there are no checkable claims, return an empty JSON array: []
"""


def extract_claims(client: Anthropic, document_text: str) -> List[Dict[str, Any]]:
    """Use Claude to pull structured, checkable claims out of document text."""
    # Guard against extremely long documents blowing the context window.
    truncated = document_text[:60000]

    response = client.messages.create(
        model=MODEL_NAME,
        max_tokens=4000,
        system=EXTRACTION_SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": f"Document text:\n\n{truncated}\n\nReturn the JSON array of claims now.",
            }
        ],
    )

    raw_text = "".join(
        block.text for block in response.content if block.type == "text"
    )
    return _safe_parse_json_array(raw_text)


def _safe_parse_json_array(raw_text: str) -> List[Dict[str, Any]]:
    """Parse a JSON array out of a model response, tolerating stray fences/text."""
    cleaned = raw_text.strip()
    cleaned = re.sub(r"^```(json)?", "", cleaned).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\[.*\]", cleaned, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                return []
        return []


# ---------------------------------------------------------------------------
# 3. CLAIM VERIFICATION (live web search via Claude's web_search tool)
# ---------------------------------------------------------------------------
VERIFICATION_SYSTEM_PROMPT = """You are a rigorous fact-checker with live web
search access. You will be given ONE factual claim extracted from a marketing
document. Search the web to check whether it is accurate TODAY.

After researching, respond with STRICT JSON only (no prose, no markdown
fences) with exactly these fields:
{
  "verdict": "<one of: Verified, Inaccurate, False>",
  "correct_fact": "<the correct, current, real fact with figure/date if applicable. If the claim was already correct, restate it briefly. If you found no evidence either way, explain that clearly here.>",
  "explanation": "<1-2 sentence explanation of why you reached this verdict>",
  "sources": ["<source name or URL>", "..."]
}

Verdict definitions:
- "Verified": the claim matches current, real, verifiable data.
- "Inaccurate": the claim is based on real underlying data but the specific
  number/date is outdated, rounded incorrectly, or otherwise off from the
  current real figure.
- "False": no credible evidence supports the claim, or it directly
  contradicts what reliable sources say, or the claim appears fabricated.

Always cite at least one source when verdict is Verified or Inaccurate.
"""


def verify_claim(client: Anthropic, claim: str) -> Dict[str, Any]:
    """Verify a single claim using Claude with the web_search tool enabled."""
    response = client.messages.create(
        model=MODEL_NAME,
        max_tokens=2000,
        system=VERIFICATION_SYSTEM_PROMPT,
        tools=[{"type": "web_search_20250305", "name": "web_search"}],
        messages=[
            {
                "role": "user",
                "content": f'Claim to verify: "{claim}"\n\nSearch the web, then respond with the JSON verdict object only.',
            }
        ],
    )

    raw_text = "".join(
        block.text for block in response.content if block.type == "text"
    )
    parsed = _safe_parse_json_object(raw_text)

    if not parsed:
        parsed = {
            "verdict": "False",
            "correct_fact": "Could not determine — model did not return a parsable verdict.",
            "explanation": "The verification step failed to produce structured output.",
            "sources": [],
        }
    return parsed


def _safe_parse_json_object(raw_text: str) -> Dict[str, Any]:
    cleaned = raw_text.strip()
    cleaned = re.sub(r"^```(json)?", "", cleaned).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                return {}
        return {}
