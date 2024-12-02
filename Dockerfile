FROM pytorch/pytorch:1.7.0-cuda11.0-cudnn8-runtime
COPY . /app
WORKDIR /app

ENV TZ=Europe
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

#ENV GPUID="0,1"

# Install some basic utilities
RUN apt-get update && apt-get install -y tzdata
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
