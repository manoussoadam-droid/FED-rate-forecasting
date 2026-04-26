# Deployment

## Systemd (Linux production server)

```bash
# 1. Copy secrets file
sudo mkdir -p /etc/fomc_tools
sudo cp .env /etc/fomc_tools/env
sudo chmod 600 /etc/fomc_tools/env

# 2. Create log directory
sudo mkdir -p /var/log/fomc_tools
sudo chown fomc:fomc /var/log/fomc_tools

# 3. Install unit files
sudo cp deploy/fomc-scheduler.service /etc/systemd/system/
sudo cp deploy/fomc-api.service       /etc/systemd/system/
sudo systemctl daemon-reload

# 4. Enable and start
sudo systemctl enable --now fomc-scheduler
sudo systemctl enable --now fomc-api

# 5. Check status
sudo systemctl status fomc-scheduler fomc-api
journalctl -u fomc-scheduler -f
```

## Requirements

- Python ≥ 3.11
- `pip install gunicorn` for the API service
- `playwright install chromium` if Bloomberg scraping is enabled

## Reverse proxy (nginx snippet)

```nginx
server {
    listen 443 ssl;
    server_name your.domain.com;

    location /api/ {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_read_timeout 60s;
    }
}
```
