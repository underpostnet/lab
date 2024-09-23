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

Required CUDA 11.8

- https://developer.nvidia.com/cuda-11-8-0-download-archive

#### Instal things separately and activating tensorflow-gpu

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

#### Instal PyTorch (GPU version compatible with CUDA version)

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
