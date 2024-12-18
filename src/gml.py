from util import index_exists
from gml_vars import categorical_vars, binary_vars, numerics_var, outcome


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import stats

df = pd.read_csv(
    "./data/soloegwg/soloegwg.csv", header=0, sep=";", decimal=".", thousands=","
)


print(df.head())
