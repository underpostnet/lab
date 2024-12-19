# import required modules
import pandas as pd
import numpy as np


# ./my-env/bin/python ./src/toDummy.py

# create dataset
df = pd.read_csv("./data/dummy.csv", header=0, sep=";", decimal=".", thousands=",")


df_dummy = np.empty((df.shape[0], 0))
df_columns = []

for column in df.columns:
    _df = pd.get_dummies(df[column], prefix=column).astype(int)  # , drop_first=True
    df_columns = df_columns + _df.columns.to_list()
    print(_df)
    df_dummy = np.concatenate((df_dummy, _df), axis=1)

df_dummy = df_dummy.tolist()

# df_dummy.insert(0, df_columns)
# df_dummy.append(df_columns)

df_dummy = pd.DataFrame(df_dummy, columns=df_columns)

for column in df_dummy.columns:
    df_dummy[column] = df_dummy[column].astype(int)

print(df_dummy)

df_dummy.to_csv(
    "./data/dummy-output.csv", sep=";", encoding="utf-8", index=False, header=True
)
