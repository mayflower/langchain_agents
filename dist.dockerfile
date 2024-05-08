# syntax=docker/dockerfile:1.4

FROM mcr.microsoft.com/devcontainers/python:1-3.11-bullseye

COPY requirements.txt /workspace/requirements.txt

WORKDIR /workspace
RUN <<EOF
    apt update
    apt install -y --no-install-recommends gcc python3-dev graphviz graphviz-dev sshpass
    python -m pip update
    python -m pip install --no-cache-dir ipykernel playwright
    python -m pip install -r requirements.txt
    python -m ipykernel install --user --name=python --display-name=Python3
EOF
COPY . /workspace