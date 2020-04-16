FROM ubuntu:18.04

ENV LANG C.UTF-8

RUN APT_INSTALL="apt-get install -y --no-install-recommends" && \
    PIP_INSTALL="python -m pip --no-cache-dir install --upgrade" && \
    rm -rf /var/lib/apt/lists/* \
           /etc/apt/sources.list.d/cuda.list \
           /etc/apt/sources.list.d/nvidia-ml.list && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
            build-essential \
            apt-utils \
            cmake \
            ca-certificates \
            wget \
            && \
    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
            software-properties-common \
            && \
        add-apt-repository ppa:deadsnakes/ppa && \
        apt-get update && \
        DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
            python3.6 \
            python3.6-dev \
            python3-distutils-extra \
            && \
        ln -s /usr/bin/python3.6 /usr/local/bin/python3 && \
        ln -s /usr/bin/python3.6 /usr/local/bin/python && \
    wget -O /tmp/get-pip.py https://bootstrap.pypa.io/get-pip.py && \
    python /tmp/get-pip.py && \
    $PIP_INSTALL setuptools && \
    ldconfig && \
    apt-get clean && \
    apt-get autoremove && \
    rm -rf /var/lib/apt/lists/* /tmp/*

WORKDIR /

COPY requirements.txt /
RUN python -m pip --no-cache-dir install --upgrade -r requirements.txt

COPY telesto /telesto
COPY *.sh /

RUN chmod +x *.sh

EXPOSE 9876

ENV PYTHONDONTWRITEBYTECODE 1

CMD [ "/start-api.sh" ]
