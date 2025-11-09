from flask import Flask, render_template, request, jsonify
import numpy as np
import io
import base64
import matplotlib
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

matplotlib.use("Agg")

app = Flask(__name__, static_folder="static", template_folder="templates")


def fig_to_base64(fig):
    buffer = io.BytesIO()
    plt.savefig(buffer, format="png", bbox_inches="tight", dpi=150)
    buffer.seek(0)
    img = base64.b64encode(buffer.getvalue()).decode()
    plt.close(fig)
    return img


def generate_regression_data(
    kind: str, n_samples: int, noise: float, random_state: int = 42
):
    rng = np.random.default_rng(random_state)
    if kind == "linear_1d":
        X = rng.uniform(-3, 3, size=(n_samples, 1))
        y = 1.5 * X[:, 0] - 0.7 + rng.normal(scale=noise, size=n_samples)
    elif kind == "sine_1d":
        X = rng.uniform(-3, 3, size=(n_samples, 1))
        y = np.sin(X[:, 0]) + rng.normal(scale=noise, size=n_samples)
    elif kind == "piecewise_1d":
        X = rng.uniform(-3, 3, size=(n_samples, 1))
        y = np.where(
            X[:, 0] < 0, -1.0 - 0.2 * X[:, 0], 1.0 + 0.5 * X[:, 0]
        ) + rng.normal(scale=noise, size=n_samples)
    elif kind == "hills_2d":
        X = rng.uniform(-2, 2, size=(n_samples, 2))
        y = (
            1.2 * np.exp(-((X[:, 0] - 1.0) ** 2) - (X[:, 1] - 0.5) ** 2)
            - 0.8 * np.exp(-((X[:, 0] + 0.8) ** 2) - (X[:, 1] + 0.8) ** 2)
            + rng.normal(scale=noise, size=n_samples)
        )
    else:
        # default to sine_1d
        X = rng.uniform(-3, 3, size=(n_samples, 1))
        y = np.sin(X[:, 0]) + rng.normal(scale=noise, size=n_samples)
    return X, y


def plot_1d_fit(X, y, x_grid, y_pred, title: str):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(X[:, 0], y, s=25, color="#1f77b4", alpha=0.7, edgecolor="k")
    ax.plot(x_grid[:, 0], y_pred, color="#d62728", linewidth=2)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    return fig


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mae = float(np.mean(np.abs(y_true - y_pred)))
    mse = float(np.mean((y_true - y_pred) ** 2))
    rmse = float(np.sqrt(mse))
    denom = np.maximum(np.abs(y_true), 1e-8)
    mape = float(np.mean(np.abs((y_true - y_pred) / denom)))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1.0 - (ss_res / ss_tot if ss_tot > 0 else 0.0)
    return {"mae": mae, "mse": mse, "rmse": rmse, "mape": mape, "r2": r2}


@app.route("/")
def index():
    return render_template("regression_linear.html")


@app.route("/rf")
def rf_index():
    return render_template("regression_rf.html")


@app.route("/linear-math")
def linear_math_index():
    return render_template("regression_linear_math.html")


@app.route("/rf-math")
def rf_math_index():
    return render_template("regression_rf_math.html")


@app.route("/api/reg-data", methods=["POST"])
def api_reg_data():
    params = request.get_json(force=True) or {}
    kind = params.get("dataset", "sine_1d")
    n_samples = int(params.get("n_samples", 200))
    noise = float(params.get("noise", 0.2))
    seed = int(params.get("random_state", 42))
    X, y = generate_regression_data(kind, n_samples, noise, seed)
    return jsonify({"X": X.tolist(), "y": [float(v) for v in y]})


@app.route("/api/linear", methods=["POST"])
def api_linear():
    try:
        params = request.get_json(force=True) or {}
        kind = params.get("dataset", "sine_1d")
        n_samples = int(params.get("n_samples", 200))
        noise = float(params.get("noise", 0.2))
        seed = int(params.get("random_state", 42))
        degree = int(params.get("degree", 1))
        fit_intercept = bool(params.get("fit_intercept", True))

        X, y = generate_regression_data(kind, n_samples, noise, seed)

        # 1D or 2D support
        is_1d = X.shape[1] == 1

        steps = [("scaler", StandardScaler())]
        if degree > 1 and is_1d:
            steps.append(
                ("poly", PolynomialFeatures(degree=degree, include_bias=False))
            )
        steps.append(("reg", LinearRegression(fit_intercept=fit_intercept)))
        model = Pipeline(steps)
        model.fit(X, y)

        # training predictions and metrics
        y_hat = model.predict(X)
        metrics = compute_metrics(y, y_hat)
        scaler = model.named_steps["scaler"]
        # Build equation (standardized feature space)
        feature_names = []
        if is_1d:
            if degree > 1:
                feature_names = ["x'"] + [
                    f"x'^{k}" for k in range(2, degree + 1)
                ]
            else:
                feature_names = ["x'"]
        else:
            feature_names = ["x1'", "x2'"]
        try:
            reg = model.named_steps["reg"]
            intercept_val = float(reg.intercept_)
            coef_list = [float(c) for c in np.ravel(reg.coef_)]
            terms = [
                f"({c:+.4f})·{name}"
                for c, name in zip(coef_list, feature_names)
            ]
            equation = (
                "ŷ = "
                + f"{intercept_val:+.4f} "
                + (" ".join(terms) if terms else "")
            )
        except Exception:
            intercept_val = 0.0
            coef_list = []
            equation = "(Equation unavailable)"

        # grid
        if is_1d:
            x_grid = np.linspace(
                X[:, 0].min() - 0.5, X[:, 0].max() + 0.5, 300
            ).reshape(-1, 1)
            y_pred = model.predict(x_grid)
            fig = plot_1d_fit(
                X, y, x_grid, y_pred, f"Linear Regression (degree={degree})"
            )
            img = fig_to_base64(fig)
            return jsonify(
                {
                    "image_1d": img,
                    "r2": float(model.score(X, y)),
                    "metrics": metrics,
                    "equation": equation,
                    "coefficients": coef_list,
                    "intercept": intercept_val,
                    "feature_names": feature_names,
                    "scaler": {
                        "mean": np.asarray(scaler.mean_).tolist(),
                        "scale": np.asarray(scaler.scale_).tolist(),
                    },
                    "degree": degree,
                    "is_1d": True,
                }
            )
        else:
            # 2D surface grid for Plotly
            x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
            y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
            gx = np.linspace(x_min, x_max, 80)
            gy = np.linspace(y_min, y_max, 80)
            xx, yy = np.meshgrid(gx, gy)
            grid = np.c_[xx.ravel(), yy.ravel()]
            zz = model.predict(grid).reshape(xx.shape)
            return jsonify(
                {
                    "x": gx.tolist(),
                    "y": gy.tolist(),
                    "z": zz.tolist(),
                    "pointsX": X[:, 0].tolist(),
                    "pointsY": X[:, 1].tolist(),
                    "pointsZ": y.tolist(),
                    "r2": float(model.score(X, y)),
                    "metrics": metrics,
                    "equation": equation,
                    "coefficients": coef_list,
                    "intercept": intercept_val,
                    "feature_names": feature_names,
                    "scaler": {
                        "mean": np.asarray(scaler.mean_).tolist(),
                        "scale": np.asarray(scaler.scale_).tolist(),
                    },
                    "degree": degree,
                    "is_1d": False,
                }
            )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/rf", methods=["POST"])
def api_rf():
    try:
        params = request.get_json(force=True) or {}
        kind = params.get("dataset", "sine_1d")
        n_samples = int(params.get("n_samples", 200))
        noise = float(params.get("noise", 0.2))
        seed = int(params.get("random_state", 42))
        n_estimators = int(params.get("n_estimators", 100))
        max_depth = params.get("max_depth", None)
        max_depth = (
            None if max_depth in (None, "null", "None", "") else int(max_depth)
        )
        min_samples_leaf = int(params.get("min_samples_leaf", 1))

        X, y = generate_regression_data(kind, n_samples, noise, seed)

        is_1d = X.shape[1] == 1

        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "rf",
                    RandomForestRegressor(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        min_samples_leaf=min_samples_leaf,
                        random_state=seed,
                    ),
                ),
            ]
        )
        model.fit(X, y)

        y_hat = model.predict(X)
        metrics = compute_metrics(y, y_hat)
        rf = model.named_steps["rf"]
        importances = rf.feature_importances_.tolist()

        if is_1d:
            x_grid = np.linspace(
                X[:, 0].min() - 0.5, X[:, 0].max() + 0.5, 300
            ).reshape(-1, 1)
            y_pred = model.predict(x_grid)
            fig = plot_1d_fit(
                X,
                y,
                x_grid,
                y_pred,
                f"Random Forest (trees={n_estimators}, depth={max_depth})",
            )
            img = fig_to_base64(fig)
            return jsonify(
                {
                    "image_1d": img,
                    "r2": float(model.score(X, y)),
                    "metrics": metrics,
                    "feature_importances": importances,
                    "is_1d": True,
                    "params": {
                        "n_estimators": n_estimators,
                        "max_depth": max_depth,
                        "min_samples_leaf": min_samples_leaf,
                    },
                }
            )
        else:
            x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
            y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
            gx = np.linspace(x_min, x_max, 80)
            gy = np.linspace(y_min, y_max, 80)
            xx, yy = np.meshgrid(gx, gy)
            grid = np.c_[xx.ravel(), yy.ravel()]
            zz = model.predict(grid).reshape(xx.shape)
            return jsonify(
                {
                    "x": gx.tolist(),
                    "y": gy.tolist(),
                    "z": zz.tolist(),
                    "pointsX": X[:, 0].tolist(),
                    "pointsY": X[:, 1].tolist(),
                    "pointsZ": y.tolist(),
                    "r2": float(model.score(X, y)),
                    "metrics": metrics,
                    "feature_importances": importances,
                    "is_1d": False,
                    "params": {
                        "n_estimators": n_estimators,
                        "max_depth": max_depth,
                        "min_samples_leaf": min_samples_leaf,
                    },
                }
            )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/linear-math")
def api_linear_math():
    explanation = {
        "title": "Linear Regression (Ordinary Least Squares)",
        "is_learnable": False,
        "has_closed_form": True,
        "objective": "minimize J(β) = (1/2) ||y - Xβ||^2",
        "solution": (
            "Normal equations: XᵀX β = Xᵀ y; "
            "if XᵀX invertible: β = (XᵀX)^{-1} Xᵀ y"
        ),
        "gradient": "∇J(β) = -Xᵀ (y - Xβ) = XᵀX β - Xᵀ y",
        "notes": [
            "With intercept, augment X with a column of ones.",
            "When XᵀX is ill-conditioned, use Ridge/L2: "
            "β = (XᵀX + λI)^{-1} Xᵀ y.",
            "Can also be learned by gradient descent on J(β) "
            "(learnable), but OLS has a closed-form.",
        ],
    }
    return jsonify(explanation)


@app.route("/api/linear-math-example")
def api_linear_math_example():
    # Simple 1D example with intercept: x=[1,2,3], y=[2.0, 2.5, 3.5]
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([2.0, 2.5, 3.5])
    x_bar = float(np.mean(x))
    y_bar = float(np.mean(y))
    s_xx = float(np.sum((x - x_bar) ** 2))
    s_xy = float(np.sum((x - x_bar) * (y - y_bar)))
    beta1 = s_xy / s_xx
    beta0 = y_bar - beta1 * x_bar
    y_hat = beta0 + beta1 * x
    metrics = compute_metrics(y, y_hat)
    steps = {
        "means": {"x_bar": x_bar, "y_bar": y_bar},
        "sums": {"S_xx": s_xx, "S_xy": s_xy},
        "formula": {
            "beta1": "S_xy / S_xx",
            "beta0": "ȳ - β1 x̄",
        },
        "values": {"beta1": float(beta1), "beta0": float(beta0)},
        "prediction_example": {"x=2.5": float(beta0 + beta1 * 2.5)},
    }
    return jsonify(
        {
            "x": x.tolist(),
            "y": y.tolist(),
            "beta0": float(beta0),
            "beta1": float(beta1),
            "y_hat": y_hat.tolist(),
            "metrics": metrics,
            "steps": steps,
        }
    )


@app.route("/api/rf-math")
def api_rf_math():
    explanation = {
        "title": "Random Forest Regressor (CART + Bagging)",
        "is_learnable": True,
        "has_closed_form": False,
        "algorithm": [
            "Train B trees on bootstrap samples of the training data.",
            "Each tree performs greedy splits to minimize impurity (MSE) "
            "in regression: impurity = mean((y - ŷ_node)^2).",
            "At each node, evaluate thresholds for features; pick split "
            "with best impurity reduction.",
            "Tree prediction is the mean of y in the leaf; forest prediction "
            "is the average across trees.",
        ],
        "impurity": (
            "For a node with targets S, prediction is mean(S); "
            "impurity = (1/|S|) ∑ (y - mean(S))^2."
        ),
        "notes": [
            "No closed-form solution; learned by greedy search.",
            "Feature importance often computed as average impurity decrease "
            "or permutation importance.",
        ],
    }
    return jsonify(explanation)


@app.route("/api/rf-math-example")
def api_rf_math_example():
    # 1D toy data: x=[0,1,2,3], y=[0,0,1,1]
    x = np.array([0.0, 1.0, 2.0, 3.0])
    y = np.array([0.0, 0.0, 1.0, 1.0])

    def mse(vals):
        mu = np.mean(vals)
        return float(np.mean((vals - mu) ** 2))

    full_mse = mse(y)
    # candidate thresholds between points: 0.5, 1.5, 2.5
    thresholds = [0.5, 1.5, 2.5]
    details = []
    for t in thresholds:
        left = y[x <= t]
        right = y[x > t]
        mse_left = mse(left)
        mse_right = mse(right)
        w = len(left) / len(y)
        split_mse = w * mse_left + (1 - w) * mse_right
        gain = full_mse - split_mse
        details.append(
            {
                "threshold": t,
                "left_mean": float(np.mean(left)),
                "right_mean": float(np.mean(right)),
                "mse_left": mse_left,
                "mse_right": mse_right,
                "weighted_mse": float(split_mse),
                "gain": float(gain),
            }
        )
    best_dict = (
        min(details, key=lambda d: d["weighted_mse"]) if details else None
    )
    # Safeguard in case no thresholds are available
    if isinstance(best_dict, dict):
        best_left = float(best_dict.get("left_mean"))
        best_right = float(best_dict.get("right_mean"))
    else:
        best_left = None
        best_right = None
    return jsonify(
        {
            "x": x.tolist(),
            "y": y.tolist(),
            "root_mse": full_mse,
            "candidates": details,
            "best_split": best_dict,
            "leaf_predictions": {
                "left": best_left,
                "right": best_right,
            },
        }
    )


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5003)
