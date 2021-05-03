#maybe upgrade to 11.03
FROM nvidia/cuda-ppc64le:10.2-cudnn8-devel-ubuntu18.04 

RUN apt-get update -y -qq
RUN apt-get upgrade -y -qq
RUN apt-get install apt-utils -y -qq
RUN apt-get install python3.7 python3.7-dev python3-pip -y -qq
RUN apt-get install gcc -y -qq

RUN python3.7 -m pip install Cython wheel
# Install correct torch dependencies for cuda 10.2 
# https://stackoverflow.com/questions/65980206/cuda-10-2-not-recognised-on-pip-installed-pytorch-1-7-1
# RUN python3.7 -m pip install torch==1.7.1+cu102 torchvision==0.8.2+cu102 torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
RUN python3.7 -m pip install torch torchvision torchaudio

RUN python3.7 -m pip install --upgrade pip
RUN python3.7 -m pip install -r requirements_server.txt