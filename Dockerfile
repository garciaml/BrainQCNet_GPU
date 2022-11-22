FROM nvidia/cuda:11.1-cudnn8-devel-ubuntu20.04

RUN sed -i '/developer\.download\.nvidia\.com\/compute\/cuda\/repos/d' /etc/apt/sources.list.d/*
RUN sed -i '/developer\.download\.nvidia\.com\/compute\/machine-learning\/repos/d' /etc/apt/sources.list.d/*

#RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/$distro/$arch/3bf863cc.pub
RUN apt-key del 7fa2af80
#RUN wget https://developer.download.nvidia.com/compute/cuda/repos/$distro/$arch/cuda-keyring_1.0-1_all.deb .
COPY cuda-keyring_1.0-1_all.deb cuda-keyring_1.0-1_all.deb
RUN dpkg -i cuda-keyring_1.0-1_all.deb

COPY . /app
WORKDIR /app

ENV TZ=Europe
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

#ENV GPUID="0,1"

# Install some basic utilities
RUN apt-get update 
RUN apt-get install -y python3 \
    python3-venv \
    make \
    tk-dev \
    tcl-dev \
    libgl1-mesa-glx \
 && apt-get install -yq libgtk2.0-dev \
 && rm -rf /var/lib/apt/lists/*

RUN make all

ENTRYPOINT ["venv/bin/python3", "run.py"]
