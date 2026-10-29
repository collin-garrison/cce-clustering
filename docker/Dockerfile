FROM nvcr.io/nvidia/pytorch:23.01-py3

USER root
WORKDIR /workspace

COPY requirements.in .
RUN python -m pip install -r requirements.in
