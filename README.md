# lando.ai

Land development research assistant: chat over uploaded PDFs with retrieval, citations, and optional chart transcription.

## Run locally

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt
streamlit run app.py
```

Set `ANTHROPIC_API_KEY` in `.streamlit/secrets.toml` (see `DEPLOYMENT.md`). Do not commit real keys.

## Deploy

See `DEPLOYMENT.md` for Docker. For **Streamlit Community Cloud**, connect this repo and add the same key under app **Secrets**.
