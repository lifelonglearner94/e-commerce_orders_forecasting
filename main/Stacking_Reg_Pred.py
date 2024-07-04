import pandas as pd
import numpy as np
import os
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
import random
from sklearn.linear_model import (Ridge, Lasso, ElasticNet,
                                  SGDRegressor, TheilSenRegressor,
                                  RANSACRegressor, PoissonRegressor, LinearRegression)
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import StackingRegressor

# Helpful functions
def create_lagged_features(df_l, value, lags):
    """
    Create lagged features for a given column in a DataFrame.

    Args:
    df_l (pd.DataFrame): Input DataFrame
    value (str): Column name to create lags for
    lags (int): Number of lags to create

    Returns:
    pd.DataFrame: DataFrame with added lagged features
    """
    for lag in range(1, lags + 1):
        df_l[f't-{lag}'] = df_l[value].shift(lag)
    df_l.dropna(inplace=True)
    return df_l

def add_cyclic_sin_cos_features(df_c, datecolumn="date"):
    """
    Add cyclic features (sin and cos) for day of year and day of week.

    Args:
    df_c (pd.DataFrame): Input DataFrame
    datecolumn (str): Name of the date column

    Returns:
    pd.DataFrame: DataFrame with added cyclic features
    """
    # Create sin and cos features for day of year
    df_c['dayofyear_sin'] = np.sin(2 * np.pi * df_c[datecolumn].dt.dayofyear/365.25)
    df_c['dayofyear_cos'] = np.cos(2 * np.pi * df_c[datecolumn].dt.dayofyear/365.25)

    # Create sin and cos features for day of week
    df_c['dayofweek_sin'] = np.sin(2 * np.pi * df_c[datecolumn].dt.dayofweek/7)
    df_c['dayofweek_cos'] = np.cos(2 * np.pi * df_c[datecolumn].dt.dayofweek/7)

    return df_c

def scale_t_columns(df):
    """
    Scale the t-columns (lagged features) using StandardScaler.

    Args:
    df (pd.DataFrame): Input DataFrame

    Returns:
    tuple: (scaled DataFrame, fitted StandardScaler)
    """
    # Identify t-columns
    t_columns = [col for col in df.columns if col.startswith('t-') and col[2:].isdigit() and 1 <= int(col[2:]) <= 14]

    # Sort t-columns to ensure they're in order
    t_columns.sort(key=lambda x: int(x[2:]))

    # Create a StandardScaler
    scaler = StandardScaler()

    # Fit and transform t-columns
    scaled_t = scaler.fit_transform(df[t_columns])

    # Create a new dataframe with scaled values
    scaled_df = df.copy()
    scaled_df[t_columns] = scaled_t

    return scaled_df, scaler

def get_random_stacking_regressor():
    """
    Create a random StackingRegressor with a subset of base models.

    Returns:
    tuple: (list of used regressor names, StackingRegressor object)
    """
    xgb_regressor = xgb.XGBRegressor(objective='reg:squarederror', seed=42)
    xgb_pipe = make_pipeline(xgb_regressor)
    regressors = [
        ('xgb', xgb_pipe),
        ('ridge', Ridge(alpha=1.0)),
        ('lasso', Lasso(alpha=0.1)),
        ('elastic_net', ElasticNet(alpha=0.1, l1_ratio=0.5)),
        ('sgd', SGDRegressor(max_iter=1000, tol=1e-3)),
        ('decision_tree', DecisionTreeRegressor()),
        ('random_forest', RandomForestRegressor()),
        ('gradient_boosting', GradientBoostingRegressor()),
        ('svr', SVR(kernel='rbf')),
        ('knn', KNeighborsRegressor()),
        ('theil_sen', TheilSenRegressor()),
        ('ransac', RANSACRegressor()),
        ('poisson', PoissonRegressor())
    ]

    num_regressors = random.randint(3, min(10, len(regressors)))

    base_models = random.sample(regressors, num_regressors)
    # Define meta-model
    meta_model = LinearRegression()

    used_regressors = [i[0] for i in base_models]
    # Create the stacking regressor
    stacking_regressor = StackingRegressor(
        estimators=base_models,
        final_estimator=meta_model,
        cv=5
    )
    return used_regressors, stacking_regressor

def cross_val_modelX(data, model):
    """
    Perform cross-validation for a given model on each warehouse's data.

    Args:
    data (pd.DataFrame): Input data
    model: Model to evaluate

    Returns:
    list: List of mean absolute percentage errors for each warehouse
    """
    result_list = []

    try:
        data = data.set_index("date")
    except:
        pass

    warehouses = data["warehouse"].unique().tolist()

    for warehouse in warehouses:
        warehouse_data = data[data["warehouse"] == warehouse]

        warehouse_data_X = warehouse_data.drop(columns=["orders", "warehouse"])
        warehouse_data_y = warehouse_data["orders"]

        final_score = abs(cross_val_score(model, warehouse_data_X, warehouse_data_y, cv=5, scoring="neg_mean_absolute_percentage_error").mean())

        result_list.append(final_score)
        print(warehouse, final_score)

    return result_list

# Main execution
data_folder = "data"
filename = "train.csv"
df = pd.read_csv(os.path.join(data_folder, filename))
df['date'] = pd.to_datetime(df['date'])
test_cols_plus_y = ["orders", "warehouse", "date", "holiday_name", "holiday", "shops_closed", "winter_school_holidays", "school_holidays", "id"]
df = df[test_cols_plus_y]

# Drop categorical columns
df = df.drop(columns=["holiday_name", "id"])

# Prepare final dataset
final_df = add_cyclic_sin_cos_features(df.copy(), datecolumn="date")
final_df = create_lagged_features(final_df.copy(), "orders", lags=14)
final_df["daterange"] = list(range(len(final_df)))

scaled_data, scaler = scale_t_columns(final_df.copy())

warehouses = scaled_data["warehouse"].unique().tolist()

score_dataframe = pd.DataFrame(columns=warehouses)

# Perform multiple iterations of model evaluation
for i in range(150):
    used_regressors, model = get_random_stacking_regressor()

    result_list = cross_val_modelX(scaled_data, model)

    result_list.append(set(used_regressors))

    new_row = pd.DataFrame([result_list], columns=[*warehouses, "reg"])
    score_dataframe = pd.concat([score_dataframe, new_row], ignore_index=True)

# Save results
score_dataframe.to_csv("scores.csv")

# Print best models for each warehouse
for warehouse in warehouses:
    print(warehouse)
    print(score_dataframe.loc[score_dataframe[warehouse].idxmin()])
    print()
