if [ $# -ne 1 ]; then
  echo 1>&2 "please specify development, staging, or production"
  exit 2
fi

aws ecr get-login-password --region us-east-1 | sudo docker login --username AWS --password-stdin 805139563725.dkr.ecr.us-east-1.amazonaws.com && \
sudo docker tag fmv_websocket_server:latest 805139563725.dkr.ecr.us-east-1.amazonaws.com/fmv_websocket_server:$1 && \
sudo docker push 805139563725.dkr.ecr.us-east-1.amazonaws.com/fmv_websocket_server:$1 && \
sudo docker logout 805139563725.dkr.ecr.us-east-1.amazonaws.com
