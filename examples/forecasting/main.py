import pandas as pd
import numpy as np
from flask import Flask, render_template, jsonify, request
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import io
import requests
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.arima.model import ARIMA
import warnings

app = Flask(__name__)

# Global variable to store data
DATA_CACHE = None


def load_and_preprocess_data():
    global DATA_CACHE
    if DATA_CACHE is not None:
        return DATA_CACHE

    url = "https://raw.githubusercontent.com/rasvob/2020-2021-DA4/master/datasets/ppnet_metar_v8_MAD.csv"
    # Use requests to get content if direct read fails (optional, but robust)
    try:
        s = requests.get(url, timeout=10).content
        df = pd.read_csv(io.StringIO(s.decode("utf-8")), sep=";", index_col=0)
    except Exception:
        # Fallback for local testing or if requests fails
        df = pd.read_csv(url, sep=";", index_col=0)

    # Filter data (2013-2018)
    df = df[df.Year < 2019].copy()

    # Test set definition
    df.loc[:, "TestSet"] = 0
    df.loc[df.Year == 2018, "TestSet"] = 1

    # DateTime index
    df.index = pd.to_datetime(df.index)

    # 1. Target Engineering: Difference from Midnight
    # Identify midnights
    df["Residual"] = df["Consumption"]
    df["Midnight_Value"] = df["Residual"].where(df.index.hour == 0).ffill()
    # Handle the first day where we might not have a midnight value yet (use first value)
    df["Midnight_Value"] = df["Midnight_Value"].fillna(df["Residual"].iloc[0])

    df["Residual_diff_from_midnight"] = df["Residual"] - df["Midnight_Value"]

    # 2. Basic Feature Engineering (that doesn't change with user params)
    # Cyclical encoding for Hour and Month
    df["Hour_sin"] = np.sin(2 * np.pi * df.index.hour / 24)
    df["Hour_cos"] = np.cos(2 * np.pi * df.index.hour / 24)
    df["Month_sin"] = np.sin(2 * np.pi * df.index.month / 12)
    df["Month_cos"] = np.cos(2 * np.pi * df.index.month / 12)

    # Domain specific
    df["IsWeekend"] = df.index.dayofweek.isin([5, 6]).astype(int)

    # Heating Season (approximate)
    # Simplified logic: If month is not 6, 7, 8 (June, July, Aug)
    df["IsHeatingSeason"] = (~df.index.month.isin([6, 7, 8])).astype(int)

    # Drop rows with NaN if any created (shouldn't be critical ones yet)
    df = df.dropna()

    DATA_CACHE = df
    return df


def create_lagged_features(df, lags, weather_lags):
    """
    Creates specific lag features based on user input.
    """
    df_features = df.copy()

    # Target Lags
    for lag in lags:
        df_features[f"Consumption_lag_{lag}"] = df_features[
            "Residual_diff_from_midnight"
        ].shift(lag)

    # Weather Lags (Temperature)
    for lag in weather_lags:
        df_features[f"Temperature_lag_{lag}"] = df_features[
            "Temperature"
        ].shift(lag)

    return df_features


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/initial_data")
def get_initial_data():
    df = load_and_preprocess_data()

    # Resample for faster plotting of the whole series (e.g., daily average)
    df_daily = df["Consumption"].resample("D").mean()

    # Perform STL Decomposition on the daily data
    # Period = 365 for yearly seasonality in daily data
    try:
        stl = STL(df_daily, period=365)
        res = stl.fit()
        trend = res.trend
        seasonal = res.seasonal
    except Exception:
        # Fallback if STL fails or statsmodels issue
        trend = df_daily.rolling(window=30, center=True).mean()
        seasonal = pd.Series(0, index=df_daily.index)

    # Calculate seasonality (simple monthly/hourly averages for bar plots)
    monthly_seasonality = df.groupby(df.index.month)["Consumption"].mean()
    hourly_seasonality = df.groupby(df.index.hour)["Consumption"].mean()

    # Create a sample dataframe for visualization of feature engineering
    # We create a version with some standard lags to show what happens
    sample_df = create_lagged_features(df.head(50), [24], [24])
    # Select a few representative columns
    cols_to_show = [
        "Consumption",
        "Residual_diff_from_midnight",
        "Hour_sin",
        "IsWeekend",
        "Consumption_lag_24",
    ]
    # Take a slice where lags are likely populated (e.g. row 25+)
    sample_slice = sample_df.iloc[25:30][cols_to_show]
    sample_data = []
    for idx, row in sample_slice.iterrows():
        record = {"Index": idx.strftime("%Y-%m-%d %H:%M")}
        for col in cols_to_show:
            val = row[col]
            if isinstance(val, float):
                val = round(val, 3)
            record[col] = val
        sample_data.append(record)

    return jsonify(
        {
            "dates": df_daily.index.strftime("%Y-%m-%d").tolist(),
            "consumption": df_daily.fillna(0).values.tolist(),
            "trend": trend.fillna(0).values.tolist(),
            "seasonal_component": seasonal.fillna(0).values.tolist(),
            "monthly_x": list(range(1, 13)),
            "monthly_y": monthly_seasonality.values.tolist(),
            "hourly_x": list(range(0, 24)),
            "hourly_y": hourly_seasonality.values.tolist(),
            "sample_data": sample_data,
        }
    )


@app.route("/api/train", methods=["POST"])
def train_model():
    data = request.json
    params = data.get("params", {})
    selected_lags = data.get("lags", [24, 48])  # Default lags

    df = load_and_preprocess_data()

    # Create features dynamically
    df_model = create_lagged_features(
        df, selected_lags, [24]
    )  # Fixed weather lag for simplicity

    # Feature selection
    # We always include basic time features + engineered lags
    features = [
        "Hour_sin",
        "Hour_cos",
        "Month_sin",
        "Month_cos",
        "IsWeekend",
        "IsHeatingSeason",
        "Temperature",
    ]
    features += [c for c in df_model.columns if "lag" in c]

    # Drop NaNs created by lagging
    df_model = df_model.dropna()

    # Split Train/Test
    # Test set is 2018 (where TestSet == 1)
    train_mask = df_model["TestSet"] == 0
    test_mask = df_model["TestSet"] == 1

    X_train = df_model.loc[train_mask, features]
    y_train = df_model.loc[train_mask, "Residual_diff_from_midnight"]
    X_test = df_model.loc[test_mask, features]
    y_test = df_model.loc[test_mask, "Residual_diff_from_midnight"]

    # Model Selection
    model_type = params.get("model_type", "rf")

    if model_type == "arima":
        # ARIMA (AutoRegressive Integrated Moving Average)
        # We treat this as ARIMAX (using X_train as exog)
        p = int(params.get("arima_p", 1))
        d = int(params.get("arima_d", 0))
        q = int(params.get("arima_q", 0))

        # To save time, we might want to limit training history if it's too large,
        # but let's try full first. Suppress convergence warnings.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            # We reset index to avoid frequency issues with missing rows
            model = ARIMA(
                endog=y_train.values, exog=X_train.values, order=(p, d, q)
            )
            res = model.fit()

            # Predict on test set using 'apply' to use actual test history (One-Step-Ahead)
            # We pass the new endog (y_test) and new exog (X_test)
            res_test = res.apply(y_test.values, exog=X_test.values)
            y_pred_diff = res_test.fittedvalues

            # Feature importance is not directly available like RF, but we can return coefs
            # We'll map coefs to feature names if possible
            # ARIMA params include AR terms + Exog terms + Variance
            # Exog terms come after AR/MA terms usually
            # This is approximate mapping for visualization
            try:
                # Filter only exog params
                exog_params = res.params[
                    len(res.param_names) - len(features) - 1 : -1
                ]  # Rough logic
                # Better: just show empty or message for now as mapping is complex
                feat_imp_vals = []
            except:
                feat_imp_vals = []

    elif model_type == "rf":
        n_estimators = int(params.get("n_estimators", 10))
        max_depth = int(params.get("max_depth", 10))
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            n_jobs=-1,
            random_state=42,
        )
        model.fit(X_train, y_train)
        y_pred_diff = model.predict(X_test)
        feat_imp_vals = model.feature_importances_.tolist()

    elif model_type == "gb":
        model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred_diff = model.predict(X_test)
        feat_imp_vals = model.feature_importances_.tolist()

    else:
        # Linear Regression
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred_diff = model.predict(X_test)
        feat_imp_vals = np.abs(
            model.coef_
        ).tolist()  # Use abs coefs as importance proxy

    # Reconstruction (Add back Midnight Value)
    # We need the midnight values for the test set
    midnight_values_test = df_model.loc[test_mask, "Midnight_Value"]
    y_pred_final = y_pred_diff + midnight_values_test
    y_true_final = df_model.loc[test_mask, "Consumption"]

    # Metrics
    mae = mean_absolute_error(y_true_final, y_pred_final)
    rmse = np.sqrt(mean_squared_error(y_true_final, y_pred_final))
    r2 = r2_score(y_true_final, y_pred_final)

    # Prepare data for plot (showing a subset of test data for clarity)
    dates = y_true_final.index.strftime("%Y-%m-%d %H:%M").tolist()

    # Handle Feature Importance Return
    # If ARIMA, we might return something else or empty
    if model_type == "arima":
        # For ARIMA, showing importance is tricky visually in the same bar chart
        # We'll just send empty
        feat_names = []
        feat_values = []
    else:
        feat_names = features
        feat_values = feat_imp_vals

    return jsonify(
        {
            "metrics": {
                "MAE": round(mae, 4),
                "RMSE": round(rmse, 4),
                "R2": round(r2, 4),
            },
            "plot_data": {
                "dates": dates,
                "y_true": y_true_final.values.tolist(),
                "y_pred": y_pred_final.tolist(),
            },
            "feature_importance": {
                "names": feat_names,
                "values": feat_values,
            },
        }
    )


if __name__ == "__main__":
    print("Starting Flask app...")
    print("Navigate to http://127.0.0.1:5000 in your browser.")
    app.run(debug=True, port=5000)
