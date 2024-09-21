import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import torch

print("torch.__version__", torch.__version__)

print("torch.cuda.is_available()", torch.cuda.is_available())
