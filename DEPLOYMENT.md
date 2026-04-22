# Land Development AI Deployment

This app can be hosted on your server using Docker, then put behind a reverse proxy for `lando.ai`.

## 1) Prepare project on server

Copy this project to your server, for example:

`/opt/landdev-ai`

Make sure these files/folders exist:

- `app.py`
- `requirements.txt`
- `Dockerfile`
- `docker-compose.yml`
- `.streamlit/secrets.toml`
- `data/inbox` (for PDFs)
- `data/index` (created automatically)

## 2) Add API key

Create/edit:

`.streamlit/secrets.toml`

```toml
ANTHROPIC_API_KEY = "your-real-key"
```

## 3) Start app with Docker

From project root:

```bash
docker compose up -d --build
```

Check logs:

```bash
docker compose logs -f landdev-ai
```

App should be available on:

`http://SERVER_IP:8501`

## 4) Point `lando.ai` to server

In your DNS provider:

- Create/verify `A` record:
  - Host: `@`
  - Value: your server public IP
- Optional:
  - Host: `www`
  - Value: your server public IP

## 5) Reverse proxy with Caddy (recommended)

Install Caddy on the server, then set `/etc/caddy/Caddyfile`:

```caddy
lando.ai, www.lando.ai {
    reverse_proxy 127.0.0.1:8501
}
```

Reload Caddy:

```bash
sudo systemctl reload caddy
```

Caddy will automatically provision HTTPS certificates.

## 6) Upload PDFs and ingest

- Copy PDFs into `data/inbox`.
- Open `https://lando.ai`.
- Click **Ingest / Update Index**.

## Troubleshooting

- If ingest does not pick up new files, verify file names and timestamps in `data/inbox`.
- If old data appears, clear `data/index/*` and ingest again.
- If site is unreachable, confirm:
  - Docker container is running
  - Port `8501` open locally
  - DNS points to correct server IP
  - Caddy/Nginx service is active
