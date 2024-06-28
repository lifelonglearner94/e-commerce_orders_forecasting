import pandas as pd
import os
from sklearn.pipeline import make_pipeline
import xgboost as xgb
import matplotlib.pyplot as plt

from sklearn.model_selection import cross_val_score

data_folder = "data"
filename = "train.csv"
df_train = pd.read_csv(os.path.join(data_folder, filename))

def create_lagged_features(df, value, lags):
    for lag in range(1, lags + 1):
        df[f't-{lag}'] = df[value].shift(lag)
    df.dropna(inplace=True)
    return df

df_train = df_train[["orders", "warehouse"]]






warehouses = df_train["warehouse"].unique().tolist()

result = pd.DataFrame({"warehouse":[],
                        "score":[],
                        "n_lag":[]})

lags = [7, 8, 9, 13]

for n_lag in lags:
    df_train_lagged = create_lagged_features(df_train, "orders", lags=n_lag)
    for warehouse in warehouses:

        warehouse_data = df_train_lagged[df_train_lagged["warehouse"] == warehouse]

        warehouse_data_X = warehouse_data.drop(columns=["orders", "warehouse"])
        warehouse_data_y = warehouse_data["orders"]

        xgb_regressor = xgb.XGBRegressor(objective='reg:squarederror', seed=42)
        xgb_pipe = make_pipeline(xgb_regressor)

        params = {
            'xgbregressor__gamma': 0,
            'xgbregressor__learning_rate': 0.01,
            'xgbregressor__max_depth': 5,
            'xgbregressor__min_child_weight': 5,
            'xgbregressor__n_estimators': 500
        }

        xgb_pipe.set_params(**params)

        final_score = abs(cross_val_score(xgb_pipe, warehouse_data_X, warehouse_data_y, cv=5, scoring="neg_mean_absolute_percentage_error").mean())

        tmp = pd.DataFrame({"warehouse":[warehouse],
                        "score":[final_score],
                        "n_lag":[n_lag]})

        result = pd.concat([result, tmp])

print(result)


# max_score_index = result['score'].idxmin()

# max_score_row = result.loc[max_score_index]

# print(max_score_row)
