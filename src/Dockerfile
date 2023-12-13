FROM 805139563725.dkr.ecr.us-east-1.amazonaws.com/lambda-stack:molyneux_22.04

EXPOSE 8765

# NOTE: the docker image uses the default settings from settings.py
# deployment scripts are responsible for environment-specific settings
ENTRYPOINT [ "/bin/bash", "-c", "source .venv/bin/activate && python fmv_websocket_server.py" ]

WORKDIR /usr/src/app/src
RUN mkdir -p ./checkpoints ../data

# trying to create virtual environment with pip causes an error but
# since we are using --system-site-packages to access lambda stack we can just use
# the system pip module (python -m pip ...)
RUN python3.10 -m venv .venv --without-pip --system-site-packages
RUN . .venv/bin/activate && python -m pip install pip-tools
COPY ./requirements.txt ./
RUN . .venv/bin/activate && pip-sync

COPY . .
RUN . .venv/bin/activate && ./build_cython.sh

ARG COMMIT_HISTORY_ARG
ENV COMMIT_HISTORY $COMMIT_HISTORY_ARG
