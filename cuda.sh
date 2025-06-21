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
conda create -y --name "${ENV_NAME}" python=3.9

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

# Installing packages sequentially.
# NOTE: It is strongly recommended to install all packages in a single 'conda install'
# command. This allows conda's dependency solver to find a compatible set of packages
# all at once, which is more efficient and less prone to conflicts. Installing
# packages one-by-one can sometimes lead to an inconsistent environment.
echo "Installing tensorflow-gpu..."
conda install -y tensorflow-gpu
echo "Installing python-dotenv..."
conda install -y python-dotenv
echo "Installing pynvml..."
conda install -y pynvml
echo "Installing sentencepiece..."
conda install -y sentencepiece
echo "Installing huggingface_hub..."
conda install -y huggingface_hub
echo "Installing transformers..."
conda install -y transformers
echo "Installing accelerate..."
conda install -y accelerate
echo "Installing beautifulsoup4..."
conda install -y beautifulsoup4
echo "Installing matplotlib..."
conda install -y matplotlib
echo "Installing keras..."
conda install -y keras
echo "Installing numpy version 1.23.4..."
conda install -y numpy==1.23.4
echo "Installing cudatoolkit from anaconda channel..."
conda install -y -c anaconda cudatoolkit
echo "Installing pytorch from pytorch channel..."
conda install -y -c pytorch pytorch
echo "Installing torchvision from pytorch channel..."
conda install -y -c pytorch torchvision
echo "Installing torchaudio from pytorch channel..."
conda install -y -c pytorch torchaudio
echo "Installing pytorch-cuda=11.8 from pytorch channel..."
conda install -y -c pytorch pytorch-cuda=11.8
echo "Installing diffusers from conda-forge channel..."
conda install -y -c conda-forge diffusers

echo -e "\n\nEnvironment '${ENV_NAME}' created and packages installed successfully."
echo "To activate it in a new terminal, first ensure you have run 'conda init <your-shell>' once, restarted your terminal, then run:"
echo "conda activate ${ENV_NAME}"
