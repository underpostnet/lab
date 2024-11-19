<!-- julia main.js args -->

#### generate project

```
]

pkg > generate MyProject
```

#### install package

```julia
using Pkg

Pkg.resolve()

```

#### docs

- https://docs.julialang.org/en/v1/

- https://www.kaggle.com/datasets/ilknuricke/neurohackinginrimages?resource=download

- https://wtclarke.github.io/mrs_nifti_standard/index.html

- https://github.com/JuliaNeuroscience/NIfTI.jl

#### install

```bash
python3 -m pip install --user -r requirements.txt
```

```bash
conda install --yes --file requirements.txt
```

```bash
conda env create -f environment.yaml
```

Required CUDA 11.8

- https://developer.nvidia.com/cuda-11-8-0-download-archive

#### install things separately and activating tensorflow-gpu

```bash
conda install -c anaconda cudatoolkit
```

```bash
conda create -n tf-gpu tensorflow-gpu
```

```bash
conda activate tf-gpu
```

```bash
conda create --name cuda_env
```

```bash
conda activate cuda_env
```

```bash
conda config --append channels conda-forge
```

```bash
conda config --append channels nvidia
```

#### install PyTorch (GPU version compatible with CUDA version)

```bash
nvcc --version
```

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
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
cd /usr/bin
ls *python*
```

install python version

```bash
sudo apt-get install python3.X
```

set python version

```bash
virtualenv --python="/usr/bin/python3.11" "/path/to/new/my-env/"
```

list apt repositories

```bash
cd /etc/apt/sources.list.d
ls -a
```
