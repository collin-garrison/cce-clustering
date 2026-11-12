FROM nvcr.io/nvidia/pytorch:23.01-py3

USER root
WORKDIR /workspace

COPY requirements.txt .
RUN python -m pip install -r requirements.txt
