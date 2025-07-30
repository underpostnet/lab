
FROM nvidia/cuda:12.9.0-base-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    wget \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1

RUN pip install --no-cache-dir tensorflow[and-cuda]==2.15.0

WORKDIR /usr/local/bin
RUN wget -O getGpusResources.sh https://raw.githubusercontent.com/apache/spark/master/examples/src/main/scripts/getGpusResources.sh \
    && chmod u+x getGpusResources.sh

WORKDIR /app

CMD ["bash"]
