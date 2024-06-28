import pandas as pd
import os
from sklearn.pipeline import make_pipeline
import xgboost as xgb
import matplotlib.pyplot as plt

data_folder = "data"
filename = "train.csv"
df_train = pd.read_csv(os.path.join(data_folder, filename))

def create_lagged_features(df, value, lags):
    for lag in range(1, lags + 1):
        df[f't-{lag}'] = df[value].shift(lag)
    df.dropna(inplace=True)
    return df

df_train = df_train[["orders", "warehouse"]]

df_train_lagged = create_lagged_features(df_train, "orders", lags=14)

filename1 = "test.csv"
df_test = pd.read_csv(os.path.join(data_folder, filename1))


def predict_for_a_warehouse(warehouse: str, df_train, df_test):

    warehouse_train = df_train[df_train["warehouse"] == warehouse]

    warehouse_train_X = warehouse_train.drop(columns=["orders", "warehouse"]).reset_index(drop=True)
    warehouse_train_y = warehouse_train["orders"]


    warehouse_test = df_test[df_test["warehouse"] == warehouse].drop(columns=["warehouse"])


    row = warehouse_train_X.iloc[-1]

    shifted_row = row.shift()

    shifted_row["t-1"] = warehouse_train_y.iloc[-1]


    new_df_with_pred = pd.concat([warehouse_train_X,
                        pd.DataFrame(shifted_row).T],
                        ignore_index=True)

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

    xgb_pipe.fit(warehouse_train_X, warehouse_train_y)

    predictions = []

    for i in range(len(warehouse_test)):

        prediction_for_the_last_row = xgb_pipe.predict(pd.DataFrame(shifted_row).T)

        predictions.append(prediction_for_the_last_row[0])

        row = new_df_with_pred.iloc[-1]

        shifted_row = row.shift()

        shifted_row["t-1"] = prediction_for_the_last_row

        new_df_with_pred = pd.concat([new_df_with_pred,
                        pd.DataFrame(shifted_row).T],
                        ignore_index=True)

    return predictions


warehouses = df_train_lagged["warehouse"].unique().tolist()

all_predictions = []

for warehouse in warehouses:

    pred = predict_for_a_warehouse(warehouse, df_train_lagged, df_test)

    all_predictions += pred


final_result = pd.concat([df_test["id"], pd.DataFrame({"orders": all_predictions})], axis=1)


final_result.to_csv("submission.csv", index=False)
