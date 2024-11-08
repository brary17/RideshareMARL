FROM nvidia/cuda:12.6.2-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt update && apt -y upgrade
RUN apt install -y software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt update \
    && apt install -y python3.11 python3.11-dev python3.11-venv python3-pip

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

WORKDIR /rideshare-sim/
RUN python3 --version

COPY ./deprecated/ .
COPY ./plotting/ .
COPY ./rideshare/ .
COPY ./utils/ .
COPY ./config.ini .
COPY ./customPPO.py .
COPY ./requirements.txt .
COPY ./run-custom.py/ .

RUN python3 -m pip install -r requirements.txt

CMD [ "sleep", "infinity" ]