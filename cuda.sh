#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
# This prevents the script from continuing and printing a success message on failure.
set -e

# Define the environment name. Use the first argument if provided, otherwise default to 'cuda_env'.
ENV_NAME="${1:-cuda_env}"

# --- Environment Reset ---
echo "Attempting to remove existing '${ENV_NAME}' to ensure a clean installation..."
conda remove --name "${ENV_NAME}" --all -y || true
echo "Previous environment removed (if it existed)."

# Create the conda environment.
conda create -y --name "${ENV_NAME}" python=3.9 pip

# The 'conda init' command is for permanently configuring your interactive shell.
# It should be run once manually, then you must close and reopen your terminal.
# The error you received happens because 'conda activate' is a shell function
# that isn't available in the script's shell session immediately after 'conda init' runs.
# To make 'conda activate' work inside this script, we source the conda hooks directly:
eval "$(conda shell.bash hook)"

# Now, we can activate the environment for the subsequent commands in this script.
conda activate "${ENV_NAME}"

# Configure the channels for the environment. The warnings you saw about
# channels already existing are harmless.
conda config --append channels defaults
conda config --append channels nvidia
conda config --append channels conda-forge

echo "Installing core GPU and scientific packages with Conda for robust dependency management..."
conda install -y \
    tensorflow-gpu \
    numpy==1.23.4 \
    anaconda::cudatoolkit \
    pytorch::pytorch pytorch::torchvision pytorch::torchaudio pytorch::pytorch-cuda=11.8

conda install -y \
    python-dotenv \
    pynvml \
    sentencepiece \
    huggingface_hub \
    transformers \
    accelerate \
    matplotlib \
    beautifulsoup4 \
    scipy \
    websocket-client \
    opencv-python \
    diffusers \
    raylib \
    backgroundremover

echo -e "\n\nEnvironment '${ENV_NAME}' created and packages installed successfully."
echo "To activate it in a new terminal, first ensure you have run 'conda init <your-shell>' once, restarted your terminal, then run:"
echo "conda activate ${ENV_NAME}"
