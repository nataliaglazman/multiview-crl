#!/bin/bash

set -euo pipefail

# Build context is this directory, not the caller's cwd: the dockerfile does
# `COPY requirements.txt .`, which only resolves from docker/.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

read -p "Enter image name: [default: ${USER}] " image_name
image_name=${image_name:-${USER}}

read -p "Enter image tag: [default: v1] " image_tag
image_tag=${image_tag:-v1}

# Create a tag or name for the image
docker_tag="aicregistry:5000/${image_name}:${image_tag}"
echo "Docker tag: ${docker_tag}"

# Build the image using your Dockerfile and arguments
echo "Building the image..."
docker build -f "${SCRIPT_DIR}/dockerfile" "${SCRIPT_DIR}" \
    --tag "${docker_tag}" \
    --build-arg USER_ID="$(id -u)" \
    --build-arg GROUP_ID="$(id -g)" \
    --build-arg USER="${USER}" \
    --progress=plain \
    --no-cache

# Push the built image to aicregistry
docker push "${docker_tag}"
