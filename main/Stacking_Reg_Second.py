import pandas as pd
import os
import numpy as np

data_folder = "data"
filename = "train.csv"
df = pd.read_csv(os.path.join(data_folder, filename))

scores_df = pd.read_csv("scores.csv")

warehouses = df["warehouse"].unique().tolist()

list_of_scores = []
for warehouse in warehouses:
    print(warehouse)
    score = scores_df.loc[scores_df[warehouse].idxmin(), [warehouse]].iloc[0]
    list_of_scores.append(score)
print(np.mean(list_of_scores))
