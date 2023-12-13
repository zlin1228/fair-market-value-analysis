COMMIT_HISTORY=$( git rev-list HEAD )
sudo docker build --build-arg COMMIT_HISTORY_ARG="$COMMIT_HISTORY" -t fmv_websocket_server -f src/Dockerfile ./src
