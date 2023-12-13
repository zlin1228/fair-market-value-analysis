#!/bin/bash

# run script from within /deepmm directory (probably not necessary)
cd /deepmm && \

# remove cron jobs (if any)
(crontab -l | grep -v -F "/deepmm/deploy" || :) | crontab - && \

docker compose -f /deepmm/docker/compose.yaml -p deepmm down || echo 'deepmm removal failed'; \
docker compose -f /deepmm/docker/health_check_compose.yaml -p deepmm_health_check down || echo 'deepmm_health_check removal failed'; \

echo 'finished'
