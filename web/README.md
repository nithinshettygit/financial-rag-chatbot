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
FastAPI CORS settings. `/v1/chat` establishes a browser session with a signed,
HttpOnly cookie; the client does not send an API key.

The API uses Redis to enforce shared per-session and per-IP limits
(`CHAT_SESSION_RATE_LIMIT_REQUESTS` and `CHAT_IP_RATE_LIMIT_REQUESTS`, default
20 and 60 requests per 60 seconds). Requests over the limit receive `429` and
`Retry-After`; the frontend retries once when the delay is at most 60 seconds.
If Redis is unavailable, admission fails closed with `503`. Chat queries must
contain non-whitespace text (up to 2,000 characters); history is capped at 12
messages with at most 2,000 characters per message. `/health` and
`/health/ready` remain unauthenticated.

## Production

See the repository's [production deployment guide](../deploy/README.md) for
the Nginx and Docker Compose setup. Never place a shared production API key in
frontend code or a `VITE_*` variable. Guardrails fail closed if the guardrail
service is unavailable.

The frontend does not persist conversation history across page reloads.

The separate administrative ingestion UI remains under `admin/app.py`; it is
not part of this user-facing frontend.
