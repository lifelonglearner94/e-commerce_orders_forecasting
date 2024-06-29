import pandas as pd
import os
from sklearn.pipeline import make_pipeline
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint

def create_lagged_features(df, value, lags):
    """
    Create lagged features for a given DataFrame column.

    Args:
    df (pd.DataFrame): Input DataFrame
    value (str): Column name for which to create lagged features
    lags (int): Number of lags to create

    Returns:
    pd.DataFrame: DataFrame with added lagged features
    """
    for lag in range(1, lags + 1):
        df[f't-{lag}'] = df[value].shift(lag)
    df.dropna(inplace=True)
    return df

def predict_for_a_warehouse(warehouse: str, df_train, df_test):
    """
    Make predictions for a specific warehouse.

    Args:
    warehouse (str): Name of the warehouse
    df_train (pd.DataFrame): Training data
    df_test (pd.DataFrame): Test data

    Returns:
    list: Predictions for the warehouse
    """
    # Filter data for the specific warehouse
    warehouse_train = df_train[df_train["warehouse"] == warehouse]
    warehouse_train_X = warehouse_train.drop(columns=["orders", "warehouse"]).reset_index(drop=True)
    warehouse_train_y = warehouse_train["orders"]
    warehouse_test = df_test[df_test["warehouse"] == warehouse].drop(columns=["warehouse"])

    # Prepare data for prediction
    row = warehouse_train_X.iloc[-1]
    shifted_row = row.shift()
    shifted_row["t-1"] = warehouse_train_y.iloc[-1]
    new_df_with_pred = pd.concat([warehouse_train_X,
                        pd.DataFrame(shifted_row).T],
                        ignore_index=True)

    # Create and tune the model
    xgb_regressor = xgb.XGBRegressor(objective='reg:squarederror', seed=42)
    xgb_pipe = make_pipeline(xgb_regressor)
    best_model = randomized_hyper_tuning(xgb_pipe, warehouse_train_X, warehouse_train_y)

    best_model.fit(warehouse_train_X, warehouse_train_y)

    # Make predictions
    predictions = []
    for i in range(len(warehouse_test)):
        prediction_for_the_last_row = best_model.predict(pd.DataFrame(shifted_row).T)
        predictions.append(prediction_for_the_last_row[0])
        row = new_df_with_pred.iloc[-1]
        shifted_row = row.shift()
        shifted_row["t-1"] = prediction_for_the_last_row
        new_df_with_pred = pd.concat([new_df_with_pred,
                        pd.DataFrame(shifted_row).T],
                        ignore_index=True)
    return predictions

def randomized_hyper_tuning(model, X, y):
    """
    Perform randomized hyperparameter tuning for the given model.

    Args:
    model: The model to tune
    X (pd.DataFrame): Features
    y (pd.Series): Target variable

    Returns:
    object: Best model found by random search
    """
    # Define the parameter distribution
    param_dist = {
        'xgbregressor__learning_rate': uniform(0.009, 0.1),
        'xgbregressor__n_estimators': randint(100, 600),
        'xgbregressor__max_depth': randint(3, 8),
        'xgbregressor__min_child_weight': randint(2, 6),
        'xgbregressor__gamma': uniform(0, 0.2),
    }

    # Set up the RandomizedSearchCV
    random_search = RandomizedSearchCV(
        model,
        param_distributions=param_dist,
        n_iter=100,  # Number of parameter settings that are sampled
        cv=5,
        scoring='neg_mean_absolute_percentage_error',
        verbose=1,
        n_jobs=-1
    )
    random_search.fit(X, y)
    print(random_search.best_score_)
    return random_search.best_estimator_

def main():
    """
    Main function to run the prediction process for all warehouses.
    """
    # Load and prepare data
    data_folder = "data"
    filename = "train.csv"
    df_train = pd.read_csv(os.path.join(data_folder, filename))
    df_train = df_train[["orders", "warehouse"]]
    df_train_lagged = create_lagged_features(df_train, "orders", lags=14)

    filename1 = "test.csv"
    df_test = pd.read_csv(os.path.join(data_folder, filename1))

    # Make predictions for each warehouse
    warehouses = df_train_lagged["warehouse"].unique().tolist()
    all_predictions = []
    for warehouse in warehouses:
        print(warehouse)
        pred = predict_for_a_warehouse(warehouse, df_train_lagged, df_test)
        all_predictions += pred

    # Create and save final results
    final_result = pd.concat([df_test["id"], pd.DataFrame({"orders": all_predictions})], axis=1)
    final_result.to_csv("submission.csv", index=False)

if __name__ == "__main__":
    main()
