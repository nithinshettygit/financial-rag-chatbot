# Financial RAG Chatbot

A prototype financial-document question-answering application. It combines
document ingestion, hybrid retrieval, reranking, and grounded answer generation.
It is a single-host deployment baseline, not a high-availability service.

## Architecture

- `web/` — React/Vite chat UI.
- `app/` — FastAPI API and RAG pipeline.
- `app/retrieval/` — Qdrant dense search, local BM25 retrieval, and Voyage AI
  reranking.
- `app/services/` — retrieval orchestration, fusion, and context construction.
- `admin/` — Streamlit document-ingestion UI and Qdrant seeding tools.
- `shared/`, `config/`, `prompts/` — shared settings, non-secret defaults, and
  YAML prompts.
- `deploy/` — Nginx configuration and deployment instructions.
- `tests/`, `evaluation/` — unit tests and RAG evaluation utilities/results.

At query time, the API applies guardrails, searches Qdrant and a local BM25
index, fuses the results, reranks candidates with Voyage AI, then generates and
checks an answer with Groq. The query and retrieved passages are sent to those
external model providers. Review their data-handling terms before using
confidential documents.

## Configuration and secrets

Non-secret shared defaults are in `config/defaults.yaml`. Credentials and
deployment-specific endpoints belong in environment variables; never commit
real credentials. For deployment, copy `.env.production.example` to
`.env.production`, fill in the required values, and follow
[`deploy/README.md`](deploy/README.md) for TLS, admin authentication, Compose,
and data-backup requirements.

## Development

Requirements: Python 3.12, Node.js 22, and Docker Engine with the Compose
plugin. The production API/admin image installs the pinned Python dependencies and
spaCy model from `requirements.txt`; `requirements.in` is the input used to
regenerate that lock file. See the deployment guide for the Linux/Python 3.12
lock-generation command.

Run the unit tests from the repository root. Tests need the following
environment values because application settings are loaded at import time;
these test-only values are not real credentials:

```powershell
$env:GROQ_API_KEY = "test-key"
$env:SESSION_HMAC_SECRET = "test-session-secret-for-local-tests"
python -m unittest discover -s tests
```

For frontend development:

```sh
cd web
npm ci
npm run dev
```

The Vite development server proxies `/v1` and `/health` to
`http://127.0.0.1:8000` by default. Set `VITE_API_PROXY_TARGET` to use another
API address. The full Compose deployment requires the secrets and files
described in the deployment guide.

## Project documentation

- [Production deployment](deploy/README.md)
- [Application system design](z-Present/APP_SYSTEM_DESIGN.md)
- [Technical report](z-Present/TECHNICAL_REPORT.md)
- [Admin ingestion report](z-Present/ADMIN_INGESTION_REPORT.md)
