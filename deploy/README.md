# Production deployment

This Compose deployment is a single-host production baseline:

- Nginx serves the built React app, terminates HTTPS, and is the only service
  publishing host ports.
- Nginx applies a coarse per-IP request/connection limit. FastAPI continues to
  authenticate browser sessions with a signed, HttpOnly cookie and enforce
  shared per-session and per-IP Redis limits.
- FastAPI and the Streamlit admin UI use separate private backend networks.
  Redis Stack (for RedisVL search) is API-only; Qdrant is shared by the API and
  admin UI. Neither database is published to the host.
- The admin UI is protected by Nginx Basic Auth and its separate app login.
- Redis, Qdrant, manifests, uploaded admin files, and model downloads use
  persistent volumes.

## Requirements and secrets

Use a Linux host with Docker Engine and the Compose plugin, a DNS name pointing
to that host, inbound TCP ports 80/443, and a TLS certificate for the DNS name.
Use a managed secret store for production secrets where available; otherwise
restrict `.env.production` to the deployment account and back it up securely.

Regenerate dependency locks with Python 3.12 on Linux so platform markers match
the runtime image. Run these from the repository root; the containers receive
only dependency files, not the environment files:

```sh
docker run --rm \
  -v "$PWD/requirements.in:/work/requirements.in:ro" \
  -v "$PWD/requirements.txt:/work/requirements.txt" \
  -w /work python:3.12-slim-bookworm sh -c \
  'python -m pip install pip-tools && pip-compile --resolver=backtracking --strip-extras -o requirements.txt requirements.in'

docker run --rm \
  -v "$PWD/requirements-eval.in:/work/requirements-eval.in:ro" \
  -v "$PWD/requirements.txt:/work/requirements.txt:ro" \
  -v "$PWD/requirements-eval.txt:/work/requirements-eval.txt" \
  -w /work python:3.12-slim-bookworm sh -c \
  'python -m pip install pip-tools && pip-compile --resolver=backtracking --strip-extras -o requirements-eval.txt requirements-eval.in'
```

1. Copy `.env.production.example` to `.env.production`. The checked-in files
   contain placeholders only; replace them before deployment.
2. Set `GROQ_API_KEY`, then generate distinct high-entropy values for
   `SESSION_HMAC_SECRET`, `REDIS_PASSWORD`, `QDRANT_API_KEY`, and
   `ADMIN_PASSWORD`. The session secret must be at least 32 bytes. Generate
   random values with:

   ```sh
   python3 -c 'import secrets; print(secrets.token_urlsafe(32))'
   ```

   `FALLBACK_GROQ_API_KEY` is optional and must be a different key if set.
   Startup checks credential presence and basic format locally; it does not
   contact credential providers or log secret values.
3. Generate a separate Nginx admin password file. Do not reuse
   `ADMIN_PASSWORD`:

   ```sh
   mkdir -p deploy/secrets deploy/certs
   docker run --rm -it httpd:2.4-alpine htpasswd -B -n admin \
     > deploy/secrets/admin.htpasswd
   chmod 700 deploy/secrets
   chmod 600 .env.production
   chmod 644 deploy/secrets/admin.htpasswd
   ```

   The password file contains only a bcrypt hash. It must be readable by the
   Nginx worker; keep its parent directory restricted to the deployment
   account.
4. Put the TLS certificate chain at `deploy/certs/fullchain.pem` and its
   private key at `deploy/certs/privkey.pem`. Use a trusted certificate
   issuer/ACME client and renew it before expiry.
5. Edit both `server_name example.com` entries in
   `deploy/nginx/default.conf` to the deployed DNS name.

The React app sends same-origin chat requests directly to `/v1/chat`. The API
issues a signed, HttpOnly session cookie; the browser does not need an API key.
FastAPI enforces shared per-session and per-IP Redis limits. If Redis is
unavailable, chat admission fails closed with `503`; cache operations degrade
to uncached requests and Redis reconnects automatically. The frontend retries
a `429` once when `Retry-After` is no more than 60 seconds.

## Start, inspect, and update

From the repository root on the deployment host:

```sh
docker compose --env-file .env.production config --quiet
docker compose --env-file .env.production up -d --build
docker compose --env-file .env.production ps
docker compose --env-file .env.production logs -f api nginx
```

Wait for `GET /health/ready` to return `{"status":"ready"}` before using chat.
The first API startup may download embedding/reranking models and take several
minutes. The Qdrant volume starts empty: use the protected admin UI to ingest
documents, or seed reviewed manifests from the admin container:

```sh
docker compose --env-file .env.production exec admin \
  python admin/seed_qdrant.py --all
```

The named `manifests` volume is initialized from the image only the first time
that volume is created. Rebuilding the image does not replace manifests already
in the volume. After indexing or changing manifests, restart every API replica
so its BM25 index reloads the files:

```sh
docker compose --env-file .env.production restart api
```

The API adds an `X-Request-ID` to each response and logs request method, path,
status, and latency without the request body. Chat spans and cache telemetry record only query hashes, not query text. A
timed-out request returns `504` while its concurrency slot remains leased until
the underlying RAG task finishes. The chat proxy timeout is bounded
at 55 seconds, above the API's 40-second total request deadline, including
queue wait.

Keep the admin route on a private network or restrict it at the host
firewall/VPN layer even though both login gates are enabled.

For updates, deploy a reviewed revision, rebuild, and check readiness:

```sh
docker compose --env-file .env.production build --pull
docker compose --env-file .env.production up -d
docker compose --env-file .env.production ps
```

Back up the `qdrant_data`, `redis_data`, and `manifests` volumes before
upgrades or destructive ingestion operations. Do not run `docker compose down
-v` against production; it removes persistent application data.

## Operational hardening

- Pin each base and service image to a reviewed digest in release deployments;
  update images through a tested change process.
- Use a host firewall so only 80/443 are internet-reachable and admin access is
  additionally limited to a VPN or trusted source IPs.
- Monitor container health, disk usage, certificate expiry, Redis availability,
  and API 429/503 rates. Configure off-host encrypted backups and test restores.
- Keep API workers at one unless the ML model memory profile and startup cost
  have been measured; Redis rate limits are shared across workers.
- The API service has no published host port and can be scaled behind Nginx
  after confirming host memory capacity:

  ```sh
  docker compose --env-file .env.production up -d --scale api=2
  ```

- This is a single-host setup, not high availability. Use managed Redis,
  Qdrant, secrets, TLS, and an external load balancer for multi-host operation.
