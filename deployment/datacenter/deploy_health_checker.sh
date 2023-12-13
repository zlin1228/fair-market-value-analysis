#!/bin/bash

if [ $# -ne 3 ]; then
  echo 1>&2 "please provide exactly 3 arguments: <port list (ex: 8810,8811,8812)> <development | production> <memorydb api key for development | production>"
  exit 2
fi

echo "$(date) - starting health checker deployment" && \
# set PATH so it is always consistent
PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/snap/bin && \
echo $PATH && \
# run script from within /deepmm directory (probably not necessary)
cd /deepmm && \

# make health_checker directory if needed
mkdir -p /deepmm/health_checker && \
# make .aws directory if needed
mkdir -p /deepmm/health_checker/.aws && \
# make logs directory if needed
mkdir -p /deepmm/health_checker/logs && \
# make www directory if needed
mkdir -p /deepmm/nginx_reverse_proxy/www && \
# download the websocket server image
AWS_CONFIG_FILE=/deepmm/health_checker/.aws/config AWS_SHARED_CREDENTIALS_FILE=/deepmm/health_checker/.aws/credentials aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 805139563725.dkr.ecr.us-east-1.amazonaws.com && \
docker pull 805139563725.dkr.ecr.us-east-1.amazonaws.com/fmv_websocket_server:$2 && \
docker logout 805139563725.dkr.ecr.us-east-1.amazonaws.com && \

# remove current deployment (if any)
# ignore errors
docker compose -f /deepmm/docker/health_check_compose.yaml -p deepmm_health_check down || : && \

# ---------- DOCKER COMPOSE ----------
# make docker directory if needed
mkdir -p /deepmm/docker && \
# create docker compose file
echo $"services:
  health_checker:
    image: 805139563725.dkr.ecr.us-east-1.amazonaws.com/fmv_websocket_server:$2
    container_name: health_checker
    # TODO: it might be possible to avoid host mode by using the macvlan network driver
    # to allow opening only the UDP multicast port
    # https://github.com/moby/libnetwork/issues/2397
    # https://github.com/moby/libnetwork/issues/552#issuecomment-1227821940
    # in the meantime, launch with host mode
    network_mode: host
    command: \"\"
    environment:
    - DEEPMM_MEMORYDB_API_KEY=$3
    entrypoint:
    - \"/bin/bash\"
    - \"-c\"
    - \"source .venv/bin/activate && python health_checker.py $1 '{ \\\"$.environment\\\": \\\"$2\\\" }'\"
    volumes:
    - /deepmm/health_checker/.aws:/root/.aws:ro
    - /deepmm/health_checker/logs:/usr/src/app/logs
    - /deepmm/nginx_reverse_proxy/www:/usr/src/app/www
" | dd of=/deepmm/docker/health_check_compose.yaml && \
chmod 400 /deepmm/docker/health_check_compose.yaml && \
# run docker compose
docker compose -f /deepmm/docker/health_check_compose.yaml -p deepmm_health_check up -d && \

# make cron directory if needed
mkdir -p /deepmm/cron && \
CRONSCRIPTPATH="/deepmm/deploy_health_checker.sh" && \
CRONREBOOTJOB="@reboot sleep 5m; $CRONSCRIPTPATH $1 $2 $3 >> /deepmm/cron/log.txt" && \
( crontab -l | grep -v -F "$CRONSCRIPTPATH" || : ; echo "$CRONREBOOTJOB" ) | crontab - && \

echo "$(date) - health checker deployment complete"
