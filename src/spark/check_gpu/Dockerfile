# Use a base image with CUDA and cuDNN runtime for Ubuntu 22.04
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Set environment variables for non-interactive installation
ENV DEBIAN_FRONTEND=noninteractive

# Install necessary packages: Java (OpenJDK), Python3, pip
RUN apt-get update
RUN apt-get install -y --no-install-recommends openjdk-11-jdk
RUN apt-get install -y --no-install-recommends python3
RUN apt-get install -y --no-install-recommends python3-pip
RUN apt-get install -y --no-install-recommends wget
RUN apt-get install -y --no-install-recommends tar
RUN apt-get install -y --no-install-recommends curl
RUN apt-get install -y --no-install-recommends gnupg
RUN apt-get install -y --no-install-recommends ca-certificates
RUN apt-get clean
RUN rm -rf /var/lib/apt/lists/*

# Set JAVA_HOME
ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64

# Download and install Spark
ARG SPARK_VERSION=3.5.1
ARG HADOOP_VERSION=3
ENV SPARK_HOME=/opt/spark

# Use archive.apache.org for more reliable downloads of specific versions
RUN wget https://archive.apache.org/dist/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz -O /tmp/spark.tgz
RUN mkdir -p ${SPARK_HOME}
RUN tar -xzf /tmp/spark.tgz -C ${SPARK_HOME} --strip-components=1
RUN rm /tmp/spark.tgz

# Add Spark bin to PATH
ENV PATH="${SPARK_HOME}/bin:${SPARK_HOME}/sbin:${PATH}"

# Install TensorFlow with CUDA dependencies
# Use tensorflow[and-cuda] to ensure all necessary CUDA/cuDNN packages are pulled
# We use --no-cache-dir to keep the image size down
RUN pip3 install --no-cache-dir tensorflow[and-cuda]

# Set working directory inside the container
WORKDIR /opt/spark/work-dir

# Copy the Python script into the working directory
COPY main_check_gpu.py .

# Set spark-submit as the explicit entrypoint for the container.
# The Spark Operator will pass its arguments (like 'driver', '--properties-file', etc.)
# directly to spark-submit. This will bypass the `exec: driver: not found` error
# that originated from `nvidia_entrypoint.sh` by ensuring spark-submit is the primary command.
ENTRYPOINT ["/opt/spark/bin/spark-submit"]
