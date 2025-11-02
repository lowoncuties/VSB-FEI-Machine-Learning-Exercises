from flask import Flask, render_template, request, jsonify
import numpy as np
import io
import base64
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from sklearn import datasets
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

matplotlib.use("Agg")

app = Flask(__name__, static_folder="static", template_folder="templates")


def fig_to_base64(fig):
    buffer = io.BytesIO()
    plt.savefig(buffer, format="png", bbox_inches="tight", dpi=150)
    buffer.seek(0)
    img = base64.b64encode(buffer.getvalue()).decode()
    plt.close(fig)
    return img


def generate_dataset(kind: str, n_samples: int, noise: float, random_state: int = 42):
    if kind == "moons":
        X, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    elif kind == "circles":
        X, y = make_circles(
            n_samples=n_samples, factor=0.5, noise=noise, random_state=random_state
        )
    elif kind == "linear":
        X, y = make_classification(
            n_samples=n_samples,
            n_features=2,
            n_redundant=0,
            n_informative=2,
            n_clusters_per_class=1,
            flip_y=min(noise, 0.2),
            class_sep=max(1.0 - noise, 0.5),
            random_state=random_state,
        )
    elif kind == "xor":
        rng = np.random.RandomState(random_state)
        X = rng.uniform(-1, 1, size=(n_samples, 2))
        y = (X[:, 0] * X[:, 1] > 0).astype(int)
        X += rng.normal(scale=noise, size=X.shape)
    else:
        # default to moons
        X, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)

    return X, y


def train_svm(X, y, kernel: str, C: float, gamma: float | str, degree: int):
    steps = [("scaler", StandardScaler())]
    svc_params = {"kernel": kernel, "C": C}
    if kernel in ("rbf", "poly", "sigmoid"):
        svc_params["gamma"] = gamma
    if kernel == "poly":
        svc_params["degree"] = degree
    steps.append(("clf", SVC(**svc_params)))
    model = Pipeline(steps)
    model.fit(X, y)
    return model


def train_ann(
    X,
    y,
    hidden_layer_sizes,
    activation: str,
    solver: str,
    max_iter: int,
    alpha: float = 0.0001,
):
    steps = [("scaler", StandardScaler())]
    clf = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        solver=solver,
        max_iter=max_iter,
        alpha=alpha,
        random_state=42,
    )
    steps.append(("clf", clf))
    model = Pipeline(steps)
    model.fit(X, y)
    return model


def ann_forward_first_hidden(model: Pipeline, X):
    scaler = model.named_steps["scaler"]
    clf: MLPClassifier = model.named_steps["clf"]
    Xs = scaler.transform(X)
    W0 = clf.coefs_[0]
    b0 = clf.intercepts_[0]
    z = Xs @ W0 + b0
    act = clf.activation
    if act == "logistic":
        a = 1 / (1 + np.exp(-z))
    elif act == "tanh":
        a = np.tanh(z)
    elif act == "relu":
        a = np.maximum(0, z)
    elif act == "identity":
        a = z
    else:
        a = z
    return a


def ann_first_hidden_z(model: Pipeline, X):
    scaler = model.named_steps["scaler"]
    clf: MLPClassifier = model.named_steps["clf"]
    Xs = scaler.transform(X)
    W0 = clf.coefs_[0]
    b0 = clf.intercepts_[0]
    z = Xs @ W0 + b0
    return z


@app.route("/api/ann-point", methods=["POST"])
def api_ann_point():
    try:
        params = request.get_json(force=True) or {}
        # dataset
        kind = params.get("dataset", "moons")
        n_samples = int(params.get("n_samples", 200))
        noise = float(params.get("noise", 0.2))
        seed = int(params.get("random_state", 42))
        X, y = generate_dataset(kind, n_samples, noise, seed)

        # model params
        hidden = params.get("hidden_layer_sizes", "5")
        if isinstance(hidden, str):
            hidden_layer_sizes = tuple(int(x.strip()) for x in hidden.split(",") if x.strip())
        else:
            hidden_layer_sizes = tuple(hidden)
        if len(hidden_layer_sizes) == 1:
            hidden_layer_sizes = hidden_layer_sizes[0]

        activation = params.get("activation", "relu")
        solver = params.get("solver", "adam")
        max_iter = int(params.get("max_iter", 300))

        # point
        point = params.get("point") or [params.get("px"), params.get("py")]
        if point is None or point[0] is None or point[1] is None:
            return jsonify({"error": "point (x,y) is required"}), 400
        px, py = float(point[0]), float(point[1])
        P = np.array([[px, py]])

        model = train_ann(X, y, hidden_layer_sizes, activation, solver, max_iter)
        scaler = model.named_steps["scaler"]
        clf: MLPClassifier = model.named_steps["clf"]

        # first layer weights
        W0 = clf.coefs_[0]  # shape (2, H)
        b0 = clf.intercepts_[0]  # shape (H,)
        Xs = scaler.transform(P)  # shape (1,2)
        z = Xs @ W0 + b0  # (1,H)
        z = z.ravel()

        # activation
        act_name = clf.activation
        if act_name == "logistic":
            a = 1 / (1 + np.exp(-z))
        elif act_name == "tanh":
            a = np.tanh(z)
        elif act_name == "relu":
            a = np.maximum(0, z)
        elif act_name == "identity":
            a = z
        else:
            a = z

        # decision / proba
        try:
            decision = float(model.decision_function(P))
        except Exception:
            proba = model.predict_proba(P)[0]
            decision = float(proba[1] - 0.5)
        pred = int(model.predict(P)[0])

        # limit neurons for payload
        H = z.shape[0]
        k = int(min(8, H))
        neurons = []
        for i in range(k):
            neurons.append({
                "w1": float(W0[0, i]),
                "w2": float(W0[1, i]),
                "b": float(b0[i]),
                "z": float(z[i]),
                "a": float(a[i])
            })

        return jsonify({
            "point": {"x": px, "y": py, "x_scaled": float(Xs[0,0]), "y_scaled": float(Xs[0,1])},
            "activation": act_name,
            "decision": decision,
            "prediction": pred,
            "neurons": neurons,
            "num_neurons": k
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def plot_2d_decision_boundary(model, X, y, title: str):
    x_min, x_max = X[:, 0].min() - 0.8, X[:, 0].max() + 0.8
    y_min, y_max = X[:, 1].min() - 0.8, X[:, 1].max() + 0.8
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300)
    )

    grid = np.c_[xx.ravel(), yy.ravel()]
    # decision_function is available in SVC; use predict_proba fallback if needed
    try:
        zz = model.decision_function(grid)
    except Exception:
        # fall back to signed distance via probabilities
        proba = model.predict_proba(grid)
        zz = proba[:, 1] - 0.5
    zz = zz.reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(6, 5))
    # filled regions
    cs = ax.contourf(xx, yy, zz, levels=20, cmap="RdBu", alpha=0.6)
    # decision boundary and margins
    ax.contour(xx, yy, zz, levels=[-1, 0, 1], colors=["k", "k", "k"], linestyles=["--", "-", "--"], linewidths=1.2)

    # scatter points
    scatter = ax.scatter(
        X[:, 0], X[:, 1], c=y, cmap="bwr", edgecolor="k", s=40, alpha=0.9
    )

    # mark support vectors (retrieve from pipeline last step)
    try:
        svc = model.named_steps["clf"]
        scaler = model.named_steps["scaler"]
        X_scaled = scaler.transform(X)
        sv_scaled = svc.support_vectors_
        # inverse transform SVs back to original space for display
        sv = scaler.inverse_transform(sv_scaled)
        ax.scatter(
            sv[:, 0], sv[:, 1], s=120, facecolors="none", edgecolors="k", linewidths=1.5, label="Support Vectors"
        )
    except Exception:
        pass

    ax.set_title(title)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.legend(loc="upper right")
    fig.colorbar(cs, ax=ax, shrink=0.8, label="decision_function")
    return fig


def plot_3d_decision_surface(model, X, y, title: str):
    x_min, x_max = X[:, 0].min() - 0.8, X[:, 0].max() + 0.8
    y_min, y_max = X[:, 1].min() - 0.8, X[:, 1].max() + 0.8
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 80), np.linspace(y_min, y_max, 80)
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    try:
        zz = model.decision_function(grid)
    except Exception:
        proba = model.predict_proba(grid)
        zz = proba[:, 1] - 0.5
    zz = zz.reshape(xx.shape)

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(xx, yy, zz, cmap="viridis", alpha=0.8, linewidth=0, antialiased=True)

    # scatter training points at z=0 plane colored by class
    colors = np.array(["#1f77b4", "#d62728"])  # blue, red
    ax.scatter(X[:, 0], X[:, 1], np.zeros_like(X[:, 0]), c=colors[y], edgecolor="k", s=40)

    # draw z=0 plane for reference
    ax.contour(xx, yy, zz, levels=[0], colors="k", linewidths=1.5, offset=0)
    ax.set_zlim(zz.min(), zz.max())
    ax.set_title(title)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("decision_function(x)")
    return fig


@app.route("/")
def index():
    return render_template("svm.html")


@app.route("/ann")
def ann_index():
    return render_template("ann.html")


@app.route("/api/svm-surface", methods=["POST"])
def api_svm_surface():
    try:
        params = request.get_json(force=True) or {}
        # dataset
        kind = params.get("dataset", "moons")
        n_samples = int(params.get("n_samples", 200))
        noise = float(params.get("noise", 0.2))
        seed = int(params.get("random_state", 42))
        X, y = generate_dataset(kind, n_samples, noise, seed)

        # model params
        kernel = params.get("kernel", "rbf")
        C = float(params.get("C", 1.0))
        gamma_mode = params.get("gamma_mode", "value")
        if gamma_mode in ("scale", "auto"):
            gamma: float | str = gamma_mode
        else:
            gamma = float(params.get("gamma", 0.5))
        degree = int(params.get("degree", 3))

        model = train_svm(X, y, kernel, C, gamma, degree)

        # grid
        x_min, x_max = X[:, 0].min() - 0.8, X[:, 0].max() + 0.8
        y_min, y_max = X[:, 1].min() - 0.8, X[:, 1].max() + 0.8
        x_lin = np.linspace(x_min, x_max, 80)
        y_lin = np.linspace(y_min, y_max, 80)
        xx, yy = np.meshgrid(x_lin, y_lin)
        grid = np.c_[xx.ravel(), yy.ravel()]
        try:
            zz = model.decision_function(grid)
        except Exception:
            proba = model.predict_proba(grid)
            zz = proba[:, 1] - 0.5
        zz = zz.reshape(xx.shape)

        # points elevation
        try:
            points_z = model.decision_function(X)
        except Exception:
            p = model.predict_proba(X)
            points_z = p[:, 1] - 0.5

        return jsonify({
            "x": x_lin.tolist(),
            "y": y_lin.tolist(),
            "z": zz.tolist(),
            "pointsX": X[:, 0].tolist(),
            "pointsY": X[:, 1].tolist(),
            "pointsZ": points_z.tolist(),
            "classes": [int(v) for v in y]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/ann-surface", methods=["POST"])
def api_ann_surface():
    try:
        params = request.get_json(force=True) or {}
        kind = params.get("dataset", "moons")
        n_samples = int(params.get("n_samples", 200))
        noise = float(params.get("noise", 0.2))
        seed = int(params.get("random_state", 42))
        X, y = generate_dataset(kind, n_samples, noise, seed)

        hidden = params.get("hidden_layer_sizes", "5")
        if isinstance(hidden, str):
            hidden_layer_sizes = tuple(int(x.strip()) for x in hidden.split(",") if x.strip())
        else:
            hidden_layer_sizes = tuple(hidden)
        if len(hidden_layer_sizes) == 1:
            hidden_layer_sizes = hidden_layer_sizes[0]

        activation = params.get("activation", "relu")
        solver = params.get("solver", "adam")
        max_iter = int(params.get("max_iter", 300))

        model = train_ann(X, y, hidden_layer_sizes, activation, solver, max_iter)

        x_min, x_max = X[:, 0].min() - 0.8, X[:, 0].max() + 0.8
        y_min, y_max = X[:, 1].min() - 0.8, X[:, 1].max() + 0.8
        x_lin = np.linspace(x_min, x_max, 80)
        y_lin = np.linspace(y_min, y_max, 80)
        xx, yy = np.meshgrid(x_lin, y_lin)
        grid = np.c_[xx.ravel(), yy.ravel()]
        try:
            zz = model.decision_function(grid)
        except Exception:
            proba = model.predict_proba(grid)
            zz = proba[:, 1] - 0.5
        zz = zz.reshape(xx.shape)

        try:
            points_z = model.decision_function(X)
        except Exception:
            p = model.predict_proba(X)
            points_z = p[:, 1] - 0.5

        return jsonify({
            "x": x_lin.tolist(),
            "y": y_lin.tolist(),
            "z": zz.tolist(),
            "pointsX": X[:, 0].tolist(),
            "pointsY": X[:, 1].tolist(),
            "pointsZ": points_z.tolist(),
            "classes": [int(v) for v in y]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/data", methods=["POST"])
def api_data():
    params = request.get_json(force=True) or {}
    kind = params.get("dataset", "moons")
    n_samples = int(params.get("n_samples", 200))
    noise = float(params.get("noise", 0.2))
    seed = int(params.get("random_state", 42))
    X, y = generate_dataset(kind, n_samples, noise, seed)
    return jsonify({"X": X.tolist(), "y": [int(v) for v in y], "bounds": {
        "x": [float(X[:,0].min()), float(X[:,0].max())],
        "y": [float(X[:,1].min()), float(X[:,1].max())]
    }})


@app.route("/api/svm", methods=["POST"])
def api_svm():
    try:
        params = request.get_json(force=True) or {}
        # dataset
        kind = params.get("dataset", "moons")
        n_samples = int(params.get("n_samples", 200))
        noise = float(params.get("noise", 0.2))
        seed = int(params.get("random_state", 42))
        X, y = generate_dataset(kind, n_samples, noise, seed)

        # model params
        kernel = params.get("kernel", "rbf")
        C = float(params.get("C", 1.0))
        gamma_mode = params.get("gamma_mode", "value")  # value | scale | auto
        if gamma_mode in ("scale", "auto"):
            gamma: float | str = gamma_mode
        else:
            gamma = float(params.get("gamma", 0.5))
        degree = int(params.get("degree", 3))

        model = train_svm(X, y, kernel, C, gamma, degree)

        # accuracy on training set
        acc = float(model.score(X, y))

        # support vectors count
        try:
            n_sv = int(model.named_steps["clf"].n_support_.sum())
        except Exception:
            n_sv = None

        title = f"SVM ({kernel}), C={C}, gamma={gamma}, degree={degree if kernel=='poly' else '-'}"
        fig2d = plot_2d_decision_boundary(model, X, y, title)
        img2d = fig_to_base64(fig2d)
        fig3d = plot_3d_decision_surface(model, X, y, title)
        img3d = fig_to_base64(fig3d)

        return jsonify({
            "image_2d": img2d,
            "image_3d": img3d,
            "accuracy": acc,
            "n_support_vectors": n_sv,
            "params": {
                "kernel": kernel,
                "C": C,
                "gamma": gamma,
                "degree": degree,
                "dataset": kind,
                "n_samples": n_samples,
                "noise": noise,
            }
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/ann", methods=["POST"])
def api_ann():
    try:
        params = request.get_json(force=True) or {}
        kind = params.get("dataset", "moons")
        n_samples = int(params.get("n_samples", 200))
        noise = float(params.get("noise", 0.2))
        seed = int(params.get("random_state", 42))
        X, y = generate_dataset(kind, n_samples, noise, seed)

        hidden = params.get("hidden_layer_sizes", "5")
        if isinstance(hidden, str):
            hidden_layer_sizes = tuple(int(x.strip()) for x in hidden.split(",") if x.strip())
        else:
            hidden_layer_sizes = tuple(hidden)
        if len(hidden_layer_sizes) == 1:
            hidden_layer_sizes = hidden_layer_sizes[0]

        activation = params.get("activation", "relu")
        solver = params.get("solver", "adam")
        max_iter = int(params.get("max_iter", 300))

        model = train_ann(X, y, hidden_layer_sizes, activation, solver, max_iter)

        acc = float(model.score(X, y))
        clf: MLPClassifier = model.named_steps["clf"]
        n_iter = int(getattr(clf, "n_iter_", 0))
        loss_val = float(getattr(clf, "loss_", np.nan))

        title = f"ANN hidden={hidden_layer_sizes}, act={activation}, solver={solver}, iter={max_iter}"
        fig2d = plot_2d_decision_boundary(model, X, y, title)
        img2d = fig_to_base64(fig2d)
        fig3d = plot_3d_decision_surface(model, X, y, title)
        img3d = fig_to_base64(fig3d)

        return jsonify({
            "image_2d": img2d,
            "image_3d": img3d,
            "accuracy": acc,
            "loss": loss_val,
            "n_iter": n_iter,
            "params": {
                "hidden_layer_sizes": hidden_layer_sizes,
                "activation": activation,
                "solver": solver,
                "max_iter": max_iter,
                "dataset": kind,
                "n_samples": n_samples,
                "noise": noise,
            }
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/ann-activations", methods=["POST"])
def api_ann_activations():
    try:
        params = request.get_json(force=True) or {}
        kind = params.get("dataset", "moons")
        n_samples = int(params.get("n_samples", 200))
        noise = float(params.get("noise", 0.2))
        seed = int(params.get("random_state", 42))
        X, y = generate_dataset(kind, n_samples, noise, seed)

        hidden = params.get("hidden_layer_sizes", "5")
        if isinstance(hidden, str):
            hidden_layer_sizes = tuple(int(x.strip()) for x in hidden.split(",") if x.strip())
        else:
            hidden_layer_sizes = tuple(hidden)
        if len(hidden_layer_sizes) == 1:
            hidden_layer_sizes = hidden_layer_sizes[0]

        activation = params.get("activation", "relu")
        solver = params.get("solver", "adam")
        max_iter = int(params.get("max_iter", 300))

        model = train_ann(X, y, hidden_layer_sizes, activation, solver, max_iter)

        x_min, x_max = X[:, 0].min() - 0.8, X[:, 0].max() + 0.8
        y_min, y_max = X[:, 1].min() - 0.8, X[:, 1].max() + 0.8
        x_lin = np.linspace(x_min, x_max, 80)
        y_lin = np.linspace(y_min, y_max, 80)
        xx, yy = np.meshgrid(x_lin, y_lin)
        grid = np.c_[xx.ravel(), yy.ravel()]

        Z = ann_first_hidden_z(model, grid)
        A = ann_forward_first_hidden(model, grid)
        H = Z.shape[1]
        k = int(min(12, H))
        activations = []
        pre_activations = []
        for i in range(k):
            activations.append(A[:, i].reshape(xx.shape).tolist())
            pre_activations.append(Z[:, i].reshape(xx.shape).tolist())

        return jsonify({
            "x": x_lin.tolist(),
            "y": y_lin.tolist(),
            "activations": activations,
            "pre_activations": pre_activations,
            "num_neurons": k,
            "activation": activation
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5002)


