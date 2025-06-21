#!/bin/bash

# Create the conda environment.
conda create -y --name cuda_env python=3.9

# The 'conda init' command is for permanently configuring your interactive shell.
# It should be run once manually, then you must close and reopen your terminal.
# The error you received happens because 'conda activate' is a shell function
# that isn't available in the script's shell session immediately after 'conda init' runs.
# To make 'conda activate' work inside this script, we source the conda hooks directly:
eval "$(conda shell.bash hook)"

# Now, we can activate the environment for the subsequent commands in this script.
conda activate cuda_env

# Configure the channels for the environment. The warnings you saw about
# channels already existing are harmless.
conda config --append channels defaults
conda config --append channels nvidia
conda config --append channels conda-forge

# Install all required packages. Combining them into a single command is more
# efficient for the dependency solver. The channels are specified to ensure
# packages are sourced correctly.
conda install -y tensorflow-gpu python-dotenv pynvml sentencepiece huggingface_hub transformers accelerate beautifulsoup4 matplotlib keras numpy==1.23.4 \
    anaconda::cudatoolkit \
    pytorch::pytorch pytorch::torchvision pytorch::torchaudio nvidia::pytorch-cuda=11.8 \
    conda-forge::diffusers

echo -e "\n\nEnvironment 'cuda_env' created and packages installed successfully."
echo "To activate it in a new terminal, first ensure you have run 'conda init <your-shell>' once, restarted your terminal, then run:"
echo "conda activate cuda_env"
