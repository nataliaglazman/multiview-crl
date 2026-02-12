FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive

ARG USER_ID
ARG GROUP_ID
ARG USER

RUN echo "Building with user: "$USER", user ID: "$USER_ID", group ID: "$GROUP_ID

RUN addgroup --gid $GROUP_ID $USER
RUN adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID $USER
WORKDIR /nfs/home/$USER

# Install Python and pip
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/python3 /usr/bin/python

# Install PyTorch with CUDA 12.1 support first
RUN pip3 install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu121

# (Optional): If you have additional files the Dockerfile needs to read, place them in the same folder. Before 
# using them in a command, explicitly copy them into the current directory e.g. arequirements.txt containing packages 
# to install using pip.

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt