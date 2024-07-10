import pandas as pd
import numpy as np
import os
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import RobustScaler
import random
from sklearn.linear_model import (Ridge, Lasso, ElasticNet,
                                  SGDRegressor, TheilSenRegressor,
                                  RANSACRegressor, PoissonRegressor, LinearRegression)
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import StackingRegressor

from scipy.stats import randint, uniform
from scipy.stats import loguniform

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
    scaler = RobustScaler()

    # Fit and transform t-columns
    scaled_t = scaler.fit_transform(df[t_columns])

    # Create a new dataframe with scaled values
    scaled_df = df.copy()
    scaled_df[t_columns] = scaled_t

    return scaled_df, scaler

def randomized_hyper_tuning(model, X, y, param_dist):

    # Set up the RandomizedSearchCV
    random_search = RandomizedSearchCV(
        model,
        param_distributions=param_dist,
        n_iter=1,  # Number of parameter settings that are sampled
        cv=5,
        scoring='neg_mean_absolute_percentage_error',
        verbose=1,
        n_jobs=-2
    )
    random_search.fit(X, y)
    #print(random_search.best_score_)

    #print(random_search.best_params_)

    return random_search.best_estimator_

def get_rand_param_dist_for_regressor(regressor: str):

    param_grid = {
    "xgb": {
        'xgbregressor__n_estimators': randint(50, 1000),
        'xgbregressor__learning_rate': loguniform(1e-3, 1),
        'xgbregressor__max_depth': randint(1, 15),
        'xgbregressor__min_child_weight': randint(1, 10),
        'xgbregressor__subsample': uniform(0.5, 0.5),
        'xgbregressor__colsample_bytree': uniform(0.5, 0.5),
    },
    "ridge": {
        'alpha': loguniform(1e-2, 1e3),
        'fit_intercept': [True, False],
        'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']
    },
    "lasso": {
        'alpha': loguniform(1e-3, 1e2),
        'fit_intercept': [True, False],
        'selection': ['cyclic', 'random']
    },
    "elastic_net": {
        'alpha': loguniform(1e-3, 1e2),
        'l1_ratio': uniform(0.1, 1),
        'fit_intercept': [True, False],
        'selection': ['cyclic', 'random']
    },
    "sgd": {
        'loss': ['squared_error', 'huber', 'epsilon_insensitive'],
        'penalty': ['l2', 'l1', 'elasticnet'],
        'alpha': loguniform(1e-5, 1),
        'learning_rate': ['constant', 'optimal', 'invscaling', 'adaptive'],
        'eta0': loguniform(1e-3, 1)
    },
    "decision_tree": {
        'max_depth': randint(1, 50),
        'min_samples_split': randint(2, 20),
        'min_samples_leaf': randint(1, 20),
        'max_features': uniform(0, 1)
    },
    "random_forest": {
        'n_estimators': randint(50, 500),
        'max_depth': randint(1, 50),
        'min_samples_split': randint(2, 20),
        'min_samples_leaf': randint(1, 20),
        'max_features': uniform(0, 1)
    },
    "gradient_boosting": {
        'n_estimators': randint(50, 500),
        'learning_rate': loguniform(1e-3, 1),
        'max_depth': randint(1, 15),
        'min_samples_split': randint(2, 20),
        'min_samples_leaf': randint(1, 20),
        'subsample': uniform(0.5, 0.5)
    },
    "svr": {
        'C': loguniform(1e-1, 1e3),
        'epsilon': loguniform(1e-3, 1),
        'gamma': loguniform(1e-4, 1e-1)
    },
    "knn": {
        'n_neighbors': randint(1, 50),
        'weights': ['uniform', 'distance'],
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
    },
    "theil_sen": {
            'max_subpopulation': randint(5000, 10000),  # engerer Bereich
            'n_subsamples': randint(50, 500),           # engerer Bereich
            'max_iter': randint(100, 500),              # engerer Bereich
            'tol': loguniform(1e-4, 1e-2)               # angepasst
        },
    "ransac": {
        'min_samples': uniform(0.1, 0.9),
        'max_trials': randint(100, 20000),
        'max_skips': randint(100, 20000),
        'stop_n_inliers': randint(100, 2000),
        'stop_score': uniform(0.8, 0.2),
        'stop_probability': uniform(0.95, 0.05)
    },
    "poisson": {
        'alpha': loguniform(1e-5, 1),
        'fit_intercept': [True, False],
        'max_iter': randint(100, 2000),
        'tol': loguniform(1e-5, 1e-2)
    }
    }

    return param_grid[regressor]


def get_random_stacking_regressor(list_of_regressors):
    """
    Create a random StackingRegressor with a subset of base models.

    Returns:
    tuple: (list of used regressor names, StackingRegressor object)
    """
    num_regressors = random.randint(3, min(10, len(list_of_regressors)))

    base_models = random.sample(list_of_regressors, num_regressors)
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


# def find_best_stacking_regressors():

#     warehouses = scaled_data["warehouse"].unique().tolist()

#     score_dataframe = pd.DataFrame(columns=warehouses)

#     # Perform multiple iterations of model evaluation
#     for i in range(150):
#         used_regressors, model = get_random_stacking_regressor()

#         result_list = cross_val_modelX(scaled_data, model)

#         result_list.append(set(used_regressors))

#         new_row = pd.DataFrame([result_list], columns=[*warehouses, "reg"])
#         score_dataframe = pd.concat([score_dataframe, new_row], ignore_index=True)

#         # Save results
#         score_dataframe.to_csv("scores.csv")

#     # Print best models for each warehouse
#     for warehouse in warehouses:
#         print(warehouse)
#         print(score_dataframe.loc[score_dataframe[warehouse].idxmin()])
#         print()

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


# TODO I maybe later want to scale per warehouse, so each warehouse data on its own
scaled_data, scaler = scale_t_columns(final_df.copy())


warehouses = scaled_data["warehouse"].unique().tolist()


xgb_regressor = xgb.XGBRegressor(objective='reg:squarederror', seed=42)
xgb_pipe = make_pipeline(xgb_regressor)
regressors_dict = {
    'xgb': xgb_pipe,
    'ridge': Ridge(alpha=1.0),
    'lasso': Lasso(alpha=0.1),
    'elastic_net': ElasticNet(alpha=0.1, l1_ratio=0.5),
    'sgd': SGDRegressor(max_iter=1000, tol=1e-3),
    'decision_tree': DecisionTreeRegressor(),
    'random_forest': RandomForestRegressor(),
    'gradient_boosting': GradientBoostingRegressor(),
    'svr': SVR(kernel='rbf'),
    'knn': KNeighborsRegressor(),
    'theil_sen': TheilSenRegressor(),
    'ransac': RANSACRegressor(),
    'poisson': PoissonRegressor()
}

best_regressors_warehouse_dict = {}

for warehouse in warehouses:
    try:
        scaled_data = scaled_data.set_index("date")
    except:
        pass

    warehouse_data = scaled_data[scaled_data["warehouse"] == warehouse]



    warehouse_data_X = warehouse_data.drop(columns=["orders", "warehouse"])
    warehouse_data_y = warehouse_data["orders"]

    list_of_best_regressors = []

    for key, model in regressors_dict.items():

        list_of_best_regressors.append(randomized_hyper_tuning(model, warehouse_data_X, warehouse_data_y, get_rand_param_dist_for_regressor(key)))

    best_regressors_warehouse_dict[warehouse] = [(name, regressor) for name, regressor in zip(regressors_dict.keys(), list_of_best_regressors)]



for warehouse in warehouses:

    used_reg, stacked_model = get_random_stacking_regressor(best_regressors_warehouse_dict[warehouse])
    # TODO: This is just creating one model, i need to do it several times and find the best!
