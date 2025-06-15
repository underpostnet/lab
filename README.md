#### docs

- https://docs.conda.io/projects/conda/en/4.6.0/index.html

- https://docs.anaconda.com/miniconda/install/#quick-command-line-install

- https://docs.julialang.org/en/v1/

- https://www.kaggle.com/datasets/ilknuricke/neurohackinginrimages?resource=download

- https://wtclarke.github.io/mrs_nifti_standard/index.html

- https://github.com/JuliaNeuroscience/NIfTI.jl

- https://developer.nvidia.com/cuda-11-8-0-download-archive

#### install things separately and activating tensorflow-gpu

```bash
conda create --name cuda_env python=3.9
```

```bash
conda activate cuda_env
```

```bash
conda install tensorflow-gpu python-dotenv pynvml sentencepiece huggingface_hub transformers accelerate beautifulsoup4 matplotlib keras numpy==1.23.4
```

```bash
conda install -c anaconda cudatoolkit
```

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

```bash
conda install -c conda-forge diffusers
```

#### Disable Conda from initializing automatically on startup in Linux

```bash
conda config --set auto_activate_base false
```

#### channels

```bash
conda config --show channels
```

```bash
conda config --append channels conda-forge
# or
conda config --add channels conda-forge
```

```bash
conda config --append channels nvidia
# or
conda config --add channels nvidia
```

```bash
conda config --append channels defaults
# or
conda config --add channels defaults
```

#### install others packages

```bash
# linux
pip install opencv-python backgroundremover
# window
conda install opencv rembg
```

#### list env

```bash
conda info --envs
# or
conda env list
```

#### check GPU version compatible with CUDA version

```bash
nvcc --version
```

```bash
sudo add-apt-repository ppa:graphics-drivers/ppa --yes
```

```bash
sudo apt install nvidia-driver-470  # or nvidia-driver-495
```

```bash
sudo apt install nvidia-driver
```

```bash
sudo sh NVIDIA-Linux-x86_64-470.42.01.run --dkms
```

```bash
sudo apt-get install bumblebee bumblebee-nvidia primus linux-headers-generic
```

```bash
nvidia-smi ; nvidia-smi --version
```

#### uv env

- https://docs.astral.sh/uv

Install options packages:

```bash
uv python pin <fix-python-version-required>
uv sync
```

### fastapi env

Run dev app:

```bash
uv run fastapi dev app/main.py --reload
```

Run app:

```bash
fastapi run --reload app/main.py
uvicorn app.main:app --reload
```

#### RHEL env

Install cuda toolkit 12.x

```bash
sudo find / -name "libdevice.10.bc" 2>/dev/null
conda install -c conda-forge tensorflow=2.18.0 cudnn
XLA_FLAGS=--xla_gpu_cuda_data_dir="/root/.conda/envs/cuda_env/nvvm/libdevice"
LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/root/.conda/envs/cuda_env/lib
CUDA_HOME="/root/.conda/envs/cuda_env"
```

```python
os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/root/.conda/envs/cuda_env"
```

Check GPU:

```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

#### export requirements

```bash
python3 -m pip freeze > requirements.txt
```

```bash
conda list -e > requirements.txt
```

```bash
conda create --name <env> --file requirements.txt
```

#### uninstall

```bash
python3 -m pip uninstall -r requirements.txt -y
```

```bash
conda remove pytorch torchvision torchaudio cudatoolkit
```

```bash
conda clean --all
```

#### remove env

```bash
conda remove -n cuda_env --all
```

#### python virtual environment

```bash
apt-get update
```

```bash
apt install python3.12-venv
```

```bash
python -m venv my-env
```

```bash
source ./my-env/bin/activate
```

```bash
./my-env/bin/pip install -r requirements.txt
```

### manage versions

config apt

```bash
apt install software-properties-common
sudo apt-get install python3-launchpadlib
add-apt-repository ppa:deadsnakes/ppa
apt update
```

list python versions

```bash
cd /usr/bin && ls *python*
```

install python version

```bash
sudo apt-get install python3.X
```

set python version

```bash
virtualenv --python="/usr/bin/python3.11" "/path/to/new/my-env/"
```

run script

```bash
./my-env/bin/python <path-to-script>
```

#### julia: generate project

```julia
]

pkg > generate MyProject
```

#### julia: install package

```julia
using Pkg

Pkg.resolve()

```
