#!/bin/bash

if [ $# -ne 3 ]; then
  echo 1>&2 "please provide exactly 3 arguments: <subdomain> <development | production> <memorydb api key for development | production>"
  exit 2
fi

echo "$(date) - starting deployment" && \
# set PATH so it is always consistent between shell runs and cron job runs
PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/snap/bin && \
echo $PATH && \
# run script from within /deepmm directory (probably not necessary)
cd /deepmm && \

# ---------- WEBSOCKET SERVER ----------
# pull image first since it can take a while and can be done while the previous deployment is still running
# make wss directory if needed
mkdir -p /deepmm/fmv_websocket_server && \
# make .aws directory if needed
mkdir -p /deepmm/fmv_websocket_server/.aws && \
# make s3-cache directory if needed
mkdir -p /deepmm/fmv_websocket_server/s3-cache && \
# clear the s3-cache directory
rm -r /deepmm/fmv_websocket_server/s3-cache/* || : && \
# make data directory if needed
mkdir -p /deepmm/fmv_websocket_server/data && \
# make logs directory if needed
mkdir -p /deepmm/fmv_websocket_server/logs && \
# download the websocket server image
AWS_CONFIG_FILE=/deepmm/fmv_websocket_server/.aws/config AWS_SHARED_CREDENTIALS_FILE=/deepmm/fmv_websocket_server/.aws/credentials aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 805139563725.dkr.ecr.us-east-1.amazonaws.com && \
docker pull 805139563725.dkr.ecr.us-east-1.amazonaws.com/fmv_websocket_server:$2 && \
docker logout 805139563725.dkr.ecr.us-east-1.amazonaws.com && \

# remove current deployment (if any)
# ignore errors
docker compose -f /deepmm/docker/compose.yaml -p deepmm down || : && \

# ---------- LETSENCRYPT ----------
# update snap and install certbot if needed
snap install core || echo "snap install core failed"; \
snap refresh core || echo "snap refresh core failed"; \
snap install --classic certbot || echo "snap install --clasic certbot failed"; \
# create letsencrypt directories if needed
mkdir -p /deepmm/letsencrypt/config-dir && \
mkdir -p /deepmm/letsencrypt/work-dir && \
mkdir -p /deepmm/letsencrypt/logs-dir && \
# create or renew ssl certificate for the specified subdomain, if needed, by
# launching a temporary standalone webserver to confirm ownership of the subdomain
# Certificate is saved at: /deepmm/letsencrypt/config-dir/live/<subdomain>.deepmm.com/fullchain.pem
# Key is saved at:         /deepmm/letsencrypt/config-dir/live/<subdomain>.deepmm.com/privkey.pem
certbot certonly --standalone \
  --email support@deepmm.com \
  --no-eff-email \
  --agree-tos \
  --keep-until-expiring \
  --config-dir /deepmm/letsencrypt/config-dir \
  --work-dir /deepmm/letsencrypt/work-dir \
  --logs-dir /deepmm/letsencrypt/logs-dir \
  -d "$1.deepmm.com" || echo "certbot failed"; \

# ---------- NGINX REVERSE PROXY ----------
# make nginx_reverse_proxy directory if needed
mkdir -p /deepmm/nginx_reverse_proxy && \
# make logs directory if needed
mkdir -p /deepmm/nginx_reverse_proxy/logs && \
# make www directory if needed
mkdir -p /deepmm/nginx_reverse_proxy/www && \
# create nginx.conf file
echo "
# use a range of ports unassigned by IANA to minimize possible port collisions
# https://www.iana.org/assignments/service-names-port-numbers/service-names-port-numbers.xhtml
# 8810-8872            Unassigned
events { }
http {
  map \$http_upgrade \$connection_upgrade {
    default upgrade;
    \"\"      close;
  }
  upstream fmv_websocket_servers {
    # assign ports in the range 8810-8839 (see above)
    server localhost:8810;
  }
  server {
    listen 443 ssl;
    server_name $1.deepmm.com;
    ssl_certificate /etc/ssl/live/$1.deepmm.com/fullchain.pem;
    ssl_certificate_key /etc/ssl/live/$1.deepmm.com/privkey.pem;
    ssl_protocols TLSv1 TLSv1.1 TLSv1.2;
    ssl_ciphers HIGH:!aNULL:!MD5;
    access_log /var/log/nginx/data-access.log combined;
    location = /health {
      default_type \"application/json\";
      alias /www/health.json;
    }
    location / {
      proxy_pass http://fmv_websocket_servers;
      proxy_http_version  1.1;
      proxy_cache_bypass  \$http_upgrade;
      proxy_set_header Upgrade           \$http_upgrade;
      proxy_set_header Connection        \$connection_upgrade;
      proxy_set_header Host              \$host;
      proxy_set_header X-Real-IP         \$remote_addr;
      proxy_set_header X-Forwarded-For   \$proxy_add_x_forwarded_for;
      proxy_set_header X-Forwarded-Proto \$scheme;
      proxy_set_header X-Forwarded-Host  \$host;
      proxy_set_header X-Forwarded-Port  \$server_port;
    }
    # provides direct-connection access to ports 8840-8869 (see above)
    location ~ ^/(88[4-6][0-9])\$ {
      proxy_pass http://127.0.0.1:\$1;
      proxy_http_version  1.1;
      proxy_cache_bypass  \$http_upgrade;
      proxy_set_header Upgrade           \$http_upgrade;
      proxy_set_header Connection        \$connection_upgrade;
      proxy_set_header Host              \$host;
      proxy_set_header X-Real-IP         \$remote_addr;
      proxy_set_header X-Forwarded-For   \$proxy_add_x_forwarded_for;
      proxy_set_header X-Forwarded-Proto \$scheme;
      proxy_set_header X-Forwarded-Host  \$host;
      proxy_set_header X-Forwarded-Port  \$server_port;
    }
  }
}
" | dd of=/deepmm/nginx_reverse_proxy/nginx.conf && \
chmod 400 /deepmm/nginx_reverse_proxy/nginx.conf && \

# ---------- DOCKER COMPOSE ----------
# make docker directory if needed
mkdir -p /deepmm/docker && \
# create docker compose file
echo $"services:
  fmv_websocket_server_1:
    image: 805139563725.dkr.ecr.us-east-1.amazonaws.com/fmv_websocket_server:$2
    container_name: fmv_websocket_server_1
    # TODO: it might be possible to avoid host mode by using the macvlan network driver
    # to allow opening only the UDP multicast port
    # https://github.com/moby/libnetwork/issues/2397
    # https://github.com/moby/libnetwork/issues/552#issuecomment-1227821940
    # in the meantime, launch with host mode and assign a unique port on the host
    network_mode: host
    command: \"\"
    environment:
    - DEEPMM_MEMORYDB_API_KEY=$3
    entrypoint:
    - \"/bin/bash\"
    - \"-c\"
    - \"source .venv/bin/activate && python fmv_websocket_server.py '{ \\\"$.environment\\\": \\\"$2\\\", \\\"$.server.fmv.port\\\": \\\"8810\\\" }'\"
    volumes:
    - /deepmm/fmv_websocket_server/.aws:/root/.aws:ro
    - /deepmm/fmv_websocket_server/s3-cache:/usr/src/app/s3-cache
    - /deepmm/fmv_websocket_server/data:/usr/src/app/data
    - /deepmm/fmv_websocket_server/logs:/usr/src/app/logs
  nginx_reverse_proxy:
    image: nginx:latest
    container_name: nginx_reverse_proxy
    # TODO: use port mapping if we are able to run without host mode
    # ports:
    # - 443:443
    network_mode: host
    volumes:
    - /deepmm/nginx_reverse_proxy/nginx.conf:/etc/nginx/nginx.conf:ro
    - /deepmm/nginx_reverse_proxy/www:/www:ro
    - /deepmm/letsencrypt/config-dir/live/:/etc/ssl/live:ro
    - /deepmm/letsencrypt/config-dir/archive/:/etc/ssl/archive:ro
    - /deepmm/nginx_reverse_proxy/logs:/var/log/nginx
" | dd of=/deepmm/docker/compose.yaml && \
chmod 400 /deepmm/docker/compose.yaml && \
# run docker compose
docker compose -f /deepmm/docker/compose.yaml -p deepmm up -d && \

# make cron directory if needed
mkdir -p /deepmm/cron && \
CRONSCRIPTPATH="/deepmm/deploy.sh" && \
# run at 7am UTC every day
CRONJOB="0 7 * * * $CRONSCRIPTPATH $1 $2 $3 >> /deepmm/cron/log.txt" && \
CRONREBOOTJOB="@reboot sleep 5m; $CRONSCRIPTPATH $1 $2 $3 >> /deepmm/cron/log.txt" && \
( crontab -l | grep -v -F "$CRONSCRIPTPATH" || : ; echo "$CRONJOB" ; echo "$CRONREBOOTJOB" ) | crontab - && \

echo "$(date) - deployment complete"
