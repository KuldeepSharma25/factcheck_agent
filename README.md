# Fact-Check Agent

A web app that automates marketing-claim verification. Upload a PDF, and the
agent extracts factual claims (statistics, dates, financial/technical
figures), checks each one against **live web data**, and flags it as:

- ✅ **Verified** — matches current, real data
- ⚠️ **Inaccurate** — based on something real, but outdated or off
- ❌ **False** — no credible evidence found, or contradicted by sources

## How it works

1. **Extract** — `pdfplumber` pulls raw text out of the uploaded PDF.
2. **Identify claims** — Google's Gemini API reads the text and returns a
   structured JSON list of specific, checkable claims (skipping vague
   marketing language with nothing to verify).
3. **Verify** — for each claim, Gemini is called again with **Google Search
   grounding** enabled, so it can check the claim against current
   information on the live web rather than relying on its training data.
4. **Report** — each claim gets a verdict, the correct/current fact, a short
   explanation, and source links. Results are shown inline and can be
   downloaded as a CSV.

## Tech stack

- **Frontend:** Streamlit
- **PDF parsing:** pdfplumber
- **Claim extraction & verification:** Google Gemini API (`gemini-3-flash-preview`)
  with built-in Google Search grounding
- **Deployment:** Streamlit Community Cloud

## Running locally

```bash
git clone <this-repo-url>
cd factcheck-app
pip install -r requirements.txt
```

Create a `.streamlit/secrets.toml` file with your API key:

```toml
GEMINI_API_KEY = "AIza..."
```

Then run:

```bash
streamlit run app.py
```

(If you don't set up secrets, the app will prompt you to paste an API key
directly into the sidebar at runtime.)

## Deployment

Deployed on **Streamlit Community Cloud**. To redeploy your own copy:

1. Push this repo to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io), connect your GitHub
   account, and select this repo + `app.py` as the entry point.
3. In the app's **Settings → Secrets**, add:
   ```toml
   GEMINI_API_KEY = "AIza..."
   ```
4. Deploy. You'll get a public URL like `https://your-app-name.streamlit.app`.

## Live App

🔗 **[Add your deployed URL here]**

## Notes / Limitations

- Verification quality depends on what the live web search surfaces — very
  recent or niche claims may return limited results.
- Large PDFs are truncated to roughly the first 60,000 characters of
  extracted text to stay within model context limits.
- Scanned PDFs without an OCR text layer will not extract any text; this app
  expects PDFs with selectable text.
