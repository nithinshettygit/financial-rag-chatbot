# Production deployment

This Compose deployment is a single-host production baseline:

- Nginx serves the built React app, terminates HTTPS, and is the only service
  publishing host ports.
- Nginx applies a coarse per-IP request/connection limit. FastAPI continues to
  authenticate `X-API-Key` and enforce a shared per-key Redis limit.
- FastAPI and the Streamlit admin UI use a private backend network. Redis
  Stack (for RedisVL search) and Qdrant are not published to the host.
- The admin UI is protected by Nginx Basic Auth and its separate app login.
- Redis, Qdrant, manifests, uploaded admin files, and model downloads use
  persistent volumes.

## Requirements and secrets

Use a Linux host with Docker Engine and the Compose plugin, a DNS name pointing
to that host, inbound TCP ports 80/443, and a TLS certificate for the DNS name.
Use a managed secret store for production secrets where available; otherwise
restrict `.env.production` to the deployment account and back it up securely.

1. Copy `.env.production.example` to `.env.production`.
2. Set `GROQ_API_KEY`, then generate distinct high-entropy values for
   `CHAT_API_KEYS`, `REDIS_PASSWORD`, `QDRANT_API_KEY`, and `ADMIN_PASSWORD`.
   API keys must be at least 32 bytes. For example:

   ```sh
   python3 -c 'import secrets; print(secrets.token_urlsafe(32))'
   ```

   `CHAT_API_KEYS` must be a JSON object; each entry is a distinct client and
   receives an independent rate-limit bucket.
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

The React app sends chat requests directly to `/v1/chat`. Production clients
must supply their own `X-API-Key`; the bundled UI accepts a per-client key in
the sidebar and keeps it in memory only for the current page. Never embed a
shared key in the static bundle or add it as a `VITE_*` setting. The Nginx
per-IP limit is only an additional guard and does not replace per-key
accounting.

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
- This is a single-host setup, not high availability. Use managed Redis,
  Qdrant, secrets, TLS, and an external load balancer for multi-host operation.
