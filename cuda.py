# https://www.educative.io/answers/how-to-resolve-torch-not-compiled-with-cuda-enabled
# python cuda.py

import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import torch

print("torch.version.cuda", torch.version.cuda)

print("torch.device('cuda')", torch.device("cuda"))

torch.device("cuda:0")

print("torch.__version__", torch.__version__)

print("torch.cuda.is_available()", torch.cuda.is_available())

# print("torch.zeros(1).cuda()", torch.zeros(1).cuda())
