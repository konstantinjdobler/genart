# syntax = docker/dockerfile:experimental

# ATTENTION: to use caching for pip installs, run export DOCKER_BUILDKIT=1 before docker build
FROM maxscha/cuda_pytorch_jupyter:10.2-cudnn8-devel-centos7-py3.8.8-small-dependencies-gcc8.4.0-pytorchv1.7.1-TEXTv0.8.1-VISIONv0.8.1

COPY requirements_server.txt .
COPY requirements.txt .
RUN python3 -m pip install --upgrade pip
# this allows us to use cahced pip installs https://stackoverflow.com/questions/58018300/using-a-pip-cache-directory-in-docker-builds
RUN --mount=type=cache,mode=0755,target=~/.cache/pip \
    python3 -m pip install -r requirements_server.txt 