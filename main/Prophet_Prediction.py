import pandas as pd
import os
from prophet import Prophet
from sklearn.model_selection import ParameterGrid
from prophet.diagnostics import cross_validation, performance_metrics

def hyper_tuning(df, param_grid):
    """
    Perform hyperparameter tuning for Prophet model using grid search.

    Args:
        df (pd.DataFrame): Input dataframe with 'ds' and 'y' columns.
        param_grid (dict): Dictionary of hyperparameters to tune.

    Returns:
        pd.Series: Best hyperparameters and corresponding MAPE score.
    """
    # Create a list of all combinations of hyperparameters
    all_params = list(ParameterGrid(param_grid))

    def cross_validation_mape(model, initial, period, horizon):
        """Calculate MAPE using cross-validation."""
        df_cv = cross_validation(model, initial=initial, period=period, horizon=horizon)
        df_p = performance_metrics(df_cv, metrics=['mape'])
        return df_p['mape'].values[0]

    list_of_result_dicts = []
    list_of_scores = []

    # Perform grid search
    for params in all_params:
        m = Prophet(
            changepoint_prior_scale=params['changepoint_prior_scale'],
            seasonality_prior_scale=params['seasonality_prior_scale'],
            holidays_prior_scale=params['holidays_prior_scale'],
            seasonality_mode=params['seasonality_mode']
        )
        m.fit(df)

        # Define the initial, period, and horizon for cross-validation
        initial = '365 days'
        period = '180 days'
        horizon = '365 days'

        # Compute MAPE using cross-validation
        mape = cross_validation_mape(m, initial, period, horizon)

        # Store the results
        list_of_result_dicts.append(params)
        list_of_scores.append(mape)

    results = pd.DataFrame({"params": list_of_result_dicts,
                            "MAPE": list_of_scores})

    # Find the best hyperparameters
    best_params = results.loc[results['MAPE'].idxmin()]
    return best_params

def main():
    """
    Main function to load data, perform hyperparameter tuning, and generate predictions.
    """
    data_folder = "data"
    filename = "train.csv"
    df = pd.read_csv(os.path.join(data_folder, filename))
    df_subset = df[["warehouse", "orders", "date"]]
    df_subset.rename(columns={"date": "ds",
                              "orders": "y"}, inplace=True)
    warehouses = df_subset["warehouse"].unique().tolist()

    filename1 = "test.csv"
    df_test = pd.read_csv(os.path.join(data_folder, filename1))
    df_test_subset = df_test[["date", "warehouse"]].rename(columns={"date": "ds"})

    all_predictions = []

    param_grid = {
    'changepoint_prior_scale': [0.2, 0.3, 0.4, 0.8],
    'seasonality_prior_scale': [0.02, 0.2, 0.4],
    'holidays_prior_scale': [0.001, 0.002],
    'seasonality_mode': ['additive', 'multiplicative']
    }

    for warehouse in warehouses:
        current_df = df_subset[df_subset["warehouse"] == warehouse].drop(columns=["warehouse"])
        current_df_test = df_test_subset[df_test_subset["warehouse"] == warehouse].drop(columns=["warehouse"])

        optimal_params = hyper_tuning(current_df, param_grid)
        model = Prophet(**optimal_params["params"])
        model.fit(current_df)
        forecast = model.predict(current_df_test)
        all_predictions += list(forecast["yhat"])

    final_result = pd.concat([df_test["id"], pd.DataFrame({"orders": all_predictions})], axis=1)
    final_result.to_csv("submission.csv", index=False)

if __name__ == "__main__":
    main()
