# FROM ubuntu:20.04
FROM nvidia/cuda:11.4.0-base-ubuntu20.04

RUN apt update -y && apt install -y python3.8 python3-pip

WORKDIR /app

COPY . .

RUN pip install -r requirements.txt

CMD ["python3", "main.py", "--mode", "test"]