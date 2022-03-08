FROM nvidia/cuda:11.1-cudnn8-devel-ubuntu20.04
COPY . /app
WORKDIR /app

ENV TZ=Europe
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

#ENV GPUID="0,1"

# Install some basic utilities
RUN apt-get update && apt-get install -y python3 \
    python3-venv \
    make \
    tk-dev \
    tcl-dev \
    libgl1-mesa-glx \
 && apt-get install -yq libgtk2.0-dev \
 && rm -rf /var/lib/apt/lists/*

RUN make all

ENTRYPOINT ["venv/bin/python3", "run.py"]
