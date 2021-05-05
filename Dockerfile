# syntax = docker/dockerfile:experimental

# ATTENTION: to use caching for pip installs, run export DOCKER_BUILDKIT=1 before docker build
FROM maxscha/cuda_pytorch_jupyter:10.2-cudnn8-devel-centos7-py3.8.8-small-dependencies-gcc8.4.0-pytorchv1.7.1-TEXTv0.8.1-VISIONv0.8.1

COPY requirements_server.txt .
COPY requirements.txt .
RUN python3 -m pip install --upgrade pip
# this allows us to use cached pip installs https://stackoverflow.com/questions/58018300/using-a-pip-cache-directory-in-docker-builds

#TODO: pandas takes ages to install even with cache because it needs to be compiled from source
# therefore we abuse layer caching in docker
# if we install additional dependencies in a new RUN command, docker can skip the pandas install step, as it has already been built previously
# The commented out RUN command is more elegant, however I have not built an image with it yet; therefore we use the old one for now
#RUN --mount=type=cache,mode=0755,target=/root/.cache/pip \
#    python3 -m pip install --cache-dir /root/.cache/pip/ -r requirements_server.txt 
RUN --mount=type=cache,mode=0755,target=~/.cache/pip \
    python3 -m pip install -r requirements_server.txt 
RUN --mount=type=cache,mode=0755,target=/root/.cache/pip \
    python3 -m pip install --cache-dir /root/.cache/pip/ torchlayers 