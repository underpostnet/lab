from util import index_exists
from glm_vars import categorical_vars, binary_vars, numerics_var, outcome

# ./my-env/bin/python ./src/glm.py
# ./my-env/bin/pip install <package>
# options: Firth logistic regression or a penalized regression.
# https://github.com/brijesh1100/LogisticRegression/tree/master/Multivariate
# https://www.nucleusbox.com/building-a-logistic-regression-model-in-python/

import pandas as pd
from sklearn.model_selection import train_test_split


df = pd.read_csv(
    "./data/soloegwg/soloegwg.csv", header=0, sep=";", decimal=".", thousands=","
)


print(df)
