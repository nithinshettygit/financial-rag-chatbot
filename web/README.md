# Financial RAG web client

This React + TypeScript frontend uses the existing FastAPI endpoints:

- `GET /health/ready`
- `POST /v1/chat`

## Local development

1. Start the API from the repository root:

   ```powershell
   uvicorn app.main:app --reload
   ```

2. In another terminal, start the frontend:

   ```powershell
   cd web
   Copy-Item .env.example .env
   npm install
   npm run dev
   ```

Vite proxies `/health` and `/v1` to `VITE_API_PROXY_TARGET` (default
`http://127.0.0.1:8000`), so local development does not require permissive
FastAPI CORS settings. `/v1/chat` requires an `X-API-Key` header. Configure
`CHAT_API_KEYS` on the API as a JSON object mapping client names to unique,
random secrets of at least 32 bytes. Generate a secret with
`python -c "import secrets; print(secrets.token_urlsafe(32))"`, then set the
API environment to `CHAT_API_KEYS='{"web":"<generated-secret>"}'`. Paste that
key into the API key field in the app sidebar; it is held in memory for the
current page only and sent over the local proxy.

The API uses Redis to enforce a shared per-key limit
(`CHAT_RATE_LIMIT_REQUESTS` per `CHAT_RATE_LIMIT_WINDOW_SECONDS`, default 20
requests per 60 seconds). Requests over the limit receive `429` and
`Retry-After`; if Redis is unavailable, chat requests fail closed with `503`.
`/health` and `/health/ready` remain unauthenticated.

## Production

See the repository's [production deployment guide](../deploy/README.md) for
the Nginx and Docker Compose setup. Never place a shared production API key in
frontend code or a `VITE_*` variable. Use distinct API keys when clients need
independent rate limits.

The frontend does not persist conversation history across page reloads.

The separate administrative ingestion UI remains under `admin/app.py`; it is
not part of this user-facing frontend.
