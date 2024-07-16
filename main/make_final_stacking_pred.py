import pandas as pd

import numpy as np

from xgboost import XGBRegressor

from sklearn.linear_model import (Ridge, Lasso, ElasticNet,
                                  SGDRegressor, TheilSenRegressor,
                                  RANSACRegressor, PoissonRegressor, LinearRegression)
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline

import os

nan = np.nan

esti_params = pd.read_csv("scores_and_estimators.csv")

# Main execution
data_folder = "data"
filename = "train.csv"
df = pd.read_csv(os.path.join(data_folder, filename))
df['date'] = pd.to_datetime(df['date'])
test_cols_plus_y = ["orders", "warehouse", "date", "holiday_name", "holiday", "shops_closed", "winter_school_holidays", "school_holidays", "id"]
df = df[test_cols_plus_y]

warehouses = df["warehouse"].unique().tolist()


dict_of_warehouse_estimator_dicts = {}

for warehouse, esti in zip(warehouses, esti_params["reg_params"]):

    dictionary = eval(esti.replace(", ...", ""))

    dict_of_warehouse_estimator_dicts[warehouse] = dictionary


# Got the best params for the stacking regressor for each warehouse !
