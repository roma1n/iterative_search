FROM ubuntu:22.04

WORKDIR /app

RUN apt update && apt upgrade && apt install -y \
    python-is-python3 \
    python3-pip

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
