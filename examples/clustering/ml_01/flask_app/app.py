# flask_app/app.py

import os
import subprocess
import base64
from io import BytesIO

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    PowerTransformer,
)
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from flask import Flask, render_template, jsonify, request
from sqlalchemy import create_engine, text

matplotlib.use("Agg")  # Use non-interactive backend

app = Flask(__name__)

# ─── locate your Scrapy project one level up ─────────────────────────────────
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCRAPY_DIR = os.path.join(BASE_DIR, "sreality")
# ──────────────────────────────────────────────────────────────────────────────

# point SQLAlchemy at the SQLite DB that Scrapy’s pipeline writes to
DB_PATH = os.path.join(SCRAPY_DIR, "apartments.db")
DB_URL = f"sqlite:///{DB_PATH}"
engine = create_engine(DB_URL, connect_args={"check_same_thread": False})


def run_spider_for_page(page: int):
    """Spawn a separate Scrapy process to crawl exactly page N."""
    subprocess.run(
        ["scrapy", "crawl", "apartments", "-a", f"page={page}"],
        cwd=SCRAPY_DIR,
        check=True,
    )


@app.route("/")
def index():
    # seed page 1 if table empty
    with engine.connect() as conn:
        cnt = conn.execute(text("SELECT COUNT(*) FROM apartments")).scalar()
    if cnt == 0:
        run_spider_for_page(1)

    # show latest 50 (page 1)
    with engine.connect() as conn:
        rows = (
            conn.execute(
                text(
                    """
            SELECT listing_id, url, title,
                   price_num, location, rooms, area_m2, image_url, price_per_m2
              FROM apartments
             ORDER BY scraped_at DESC
             LIMIT 50
        """
                )
            )
            .mappings()
            .all()
        )
    listings = [dict(r) for r in rows]
    return render_template("index.html", listings=listings)


@app.route("/api/load_page")
def load_page():
    page = int(request.args.get("page", 1))

    # run Scrapy to fetch & insert page N
    run_spider_for_page(page)

    # now grab that same batch of 50 from SQLite by ascending scraped_at
    offset = 50 * (page - 1)
    with engine.connect() as conn:
        rows = (
            conn.execute(
                text(
                    """
            SELECT listing_id, url, title,
                   price_num, location, rooms, area_m2, image_url, price_per_m2
              FROM apartments
             ORDER BY scraped_at ASC
             LIMIT 50 OFFSET :offset
        """
                ),
                {"offset": offset},
            )
            .mappings()
            .all()
        )

    # return all 50 (or fewer if you’re off the end)
    return jsonify([dict(r) for r in rows])


@app.route("/analytics")
def analytics():
    # pull area and price_per_m2
    with engine.connect() as conn:
        rows = (
            conn.execute(
                text(
                    """
            SELECT listing_id, url, title, image_url, price_num,
                   location, rooms, area_m2, price_per_m2
              FROM apartments
             WHERE area_m2 > 0
               AND price_per_m2 IS NOT NULL
        """
                )
            )
            .mappings()
            .all()
        )

    records = [dict(r) for r in rows]
    return render_template("analytics.html", records=records)


@app.route("/api/analytics_data")
def analytics_data():
    # pull the same columns as /analytics, but send as JSON
    with engine.connect() as conn:
        rows = (
            conn.execute(
                text(
                    """
            SELECT listing_id, url, title, image_url, price_num, 
                   location, rooms, area_m2, price_per_m2
              FROM apartments
             WHERE area_m2 > 0
               AND price_per_m2 >= 1
        """
                )
            )
            .mappings()
            .all()
        )
    records = [dict(r) for r in rows]
    return jsonify(records=records)


@app.route("/clustering")
def clustering():
    """Main clustering page"""
    return render_template("clustering.html")


@app.route("/clustering_viz")
def clustering_viz():
    """Interactive clustering visualization page"""
    return render_template("clustering_viz.html")


@app.route("/api/clustering_data")
def clustering_data():
    """Get data for clustering analysis"""
    with engine.connect() as conn:
        rows = (
            conn.execute(
                text(
                    """
            SELECT listing_id, url, title, image_url, price_num, 
                   location, rooms, area_m2, price_per_m2
              FROM apartments
             WHERE area_m2 > 0
               AND price_per_m2 >= 1
               AND price_num > 0
               AND rooms > 0
        """
                )
            )
            .mappings()
            .all()
        )

    records = [dict(r) for r in rows]
    df = pd.DataFrame(records)

    # Get available numeric features
    numeric_features = ["price_num", "rooms", "area_m2", "price_per_m2"]
    feature_stats = {}

    for feature in numeric_features:
        if feature in df.columns:
            values = df[feature].dropna()
            feature_stats[feature] = {
                "min": float(values.min()),
                "max": float(values.max()),
                "mean": float(values.mean()),
                "std": float(values.std()),
                "count": len(values),
            }

    return jsonify(
        {
            "records": records,
            "feature_stats": feature_stats,
            "numeric_features": numeric_features,
        }
    )


@app.route("/api/clustering_histogram")
def clustering_histogram():
    """Generate histogram for selected features"""
    with engine.connect() as conn:
        rows = (
            conn.execute(
                text(
                    """
            SELECT price_num, rooms, area_m2, price_per_m2
              FROM apartments
             WHERE area_m2 > 0
               AND price_per_m2 >= 1
               AND price_num > 0
               AND rooms > 0
        """
                )
            )
            .mappings()
            .all()
        )

    df = pd.DataFrame(rows)

    # Create histogram
    plt.figure(figsize=(12, 8))
    plt.clf()  # Clear the figure to avoid conflicts
    axes = plt.subplots(2, 2)[1].flatten()

    feature_names = {
        "price_num": "Price (CZK)",
        "rooms": "Number of Rooms",
        "area_m2": "Size (m²)",
        "price_per_m2": "Price per m² (CZK)",
    }

    for i, feature in enumerate(
        ["price_num", "rooms", "area_m2", "price_per_m2"]
    ):
        if feature in df.columns:
            data = df[feature].dropna()
            axes[i].hist(data, bins=30, alpha=0.7, edgecolor="black")
            axes[i].set_title(feature_names[feature])
            axes[i].set_xlabel(feature_names[feature])
            axes[i].set_ylabel("Frekvence")

    plt.tight_layout()

    # Convert to base64
    img_buffer = BytesIO()
    plt.savefig(img_buffer, format="png", dpi=150, bbox_inches="tight")
    img_buffer.seek(0)
    img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
    plt.close()

    return jsonify({"image": img_base64})


@app.route("/api/cluster_analysis", methods=["POST"])
def cluster_analysis():
    """Perform silhouette, elbow, or dendrogram analysis"""
    data = request.get_json()

    features = data.get("features", [])
    scaler_config = data.get("scaler_config", {})
    analysis_type = data.get(
        "analysis_type", "silhouette"
    )  # silhouette, elbow, dendrogram
    algorithm_params = data.get(
        "algorithm_params", {}
    )  # For dendrogram linkage method

    # Get data
    with engine.connect() as conn:
        rows = (
            conn.execute(
                text(
                    """
            SELECT price_num, rooms, area_m2, price_per_m2
              FROM apartments
             WHERE area_m2 > 0
               AND price_per_m2 >= 1
               AND price_num > 0
               AND rooms > 0
        """
                )
            )
            .mappings()
            .all()
        )

    df = pd.DataFrame(rows)

    if not features or len(features) < 2:
        return jsonify({"error": "Select at least 2 features"})

    # Prepare data
    X = df[features].dropna()

    if len(X) < 2:
        return jsonify({"error": "Insufficient data for analysis"})

    # Apply per-feature scaling
    X_scaled = X.copy()

    scalers = {
        "standard": StandardScaler(),
        "minmax": MinMaxScaler(),
        "robust": RobustScaler(),
        "power": PowerTransformer(method="yeo-johnson", standardize=True),
    }

    default_scaler = "standard"

    for feature in features:
        scaler_type = scaler_config.get(feature, default_scaler)
        scaler = scalers.get(scaler_type, scalers[default_scaler])

        feature_data = X[[feature]].values
        try:
            scaled_feature = scaler.fit_transform(feature_data)
            X_scaled[feature] = scaled_feature.flatten()
        except Exception as e:
            print(
                f"Warning: Failed to apply {scaler_type} to {feature}, using standard scaler: {e}"
            )
            scaler = scalers["standard"]
            scaled_feature = scaler.fit_transform(feature_data)
            X_scaled[feature] = scaled_feature.flatten()

    if analysis_type == "silhouette":
        return perform_silhouette_analysis(X_scaled)
    elif analysis_type == "elbow":
        return perform_elbow_analysis(X_scaled)
    elif analysis_type == "dendrogram":
        linkage_method = algorithm_params.get("linkage", "ward")
        return perform_dendrogram_analysis(X_scaled, linkage_method)
    else:
        return jsonify({"error": "Invalid analysis type"})


def perform_silhouette_analysis(X_scaled):
    """Perform silhouette analysis for different numbers of clusters"""
    k_range = range(2, min(11, len(X_scaled) // 2))
    silhouette_scores = []

    for k in k_range:
        try:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(X_scaled)
            silhouette_avg = silhouette_score(X_scaled, cluster_labels)
            silhouette_scores.append(silhouette_avg)
        except:
            silhouette_scores.append(0)

    # Create silhouette plot
    plt.figure(figsize=(10, 6))
    plt.clf()  # Clear the figure to avoid conflicts
    plt.plot(k_range, silhouette_scores, "bo-", linewidth=2, markersize=8)
    plt.xlabel("Number of clusters")
    plt.ylabel("Silhouette score")
    plt.title("Silhouette Analysis for Optimal Number of Clusters")
    plt.grid(True, alpha=0.3)

    # Highlight best k
    best_k = k_range[np.argmax(silhouette_scores)]
    best_score = max(silhouette_scores)
    plt.axvline(
        x=best_k,
        color="red",
        linestyle="--",
        alpha=0.7,
        label=f"Best k={best_k}",
    )
    plt.legend()

    # Convert to base64
    img_buffer = BytesIO()
    plt.savefig(img_buffer, format="png", dpi=150, bbox_inches="tight")
    img_buffer.seek(0)
    img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
    plt.close()

    return jsonify(
        {
            "image": img_base64,
            "best_k": int(best_k),
            "best_score": float(best_score),
            "scores": {
                str(k): float(score)
                for k, score in zip(k_range, silhouette_scores)
            },
        }
    )


def perform_elbow_analysis(X_scaled):
    """Perform elbow method analysis for different numbers of clusters"""
    k_range = range(1, min(11, len(X_scaled) // 2))
    inertias = []

    for k in k_range:
        try:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X_scaled)
            inertias.append(kmeans.inertia_)
        except:
            inertias.append(0)

    # Calculate elbow point (simplified method)
    if len(inertias) > 2:
        # Calculate second derivative to find elbow
        second_deriv = np.diff(inertias, 2)
        elbow_idx = np.argmax(second_deriv) + 2  # +2 because of double diff
        elbow_k = (
            k_range[elbow_idx] if elbow_idx < len(k_range) else k_range[-1]
        )
    else:
        elbow_k = k_range[1] if len(k_range) > 1 else k_range[0]

    # Create elbow plot
    plt.figure(figsize=(10, 6))
    plt.clf()  # Clear the figure to avoid conflicts
    plt.plot(k_range, inertias, "bo-", linewidth=2, markersize=8)
    plt.xlabel("Number of clusters")
    plt.ylabel("Inertia (Within-cluster sum of squares)")
    plt.title("Elbow Method for Optimal Number of Clusters")
    plt.grid(True, alpha=0.3)

    # Highlight elbow point
    plt.axvline(
        x=elbow_k,
        color="red",
        linestyle="--",
        alpha=0.7,
        label=f"Elbow k={elbow_k}",
    )
    plt.legend()

    # Convert to base64
    img_buffer = BytesIO()
    plt.savefig(img_buffer, format="png", dpi=150, bbox_inches="tight")
    img_buffer.seek(0)
    img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
    plt.close()

    return jsonify(
        {
            "image": img_base64,
            "elbow_k": int(elbow_k),
            "inertias": {
                str(k): float(inertia) for k, inertia in zip(k_range, inertias)
            },
        }
    )


def perform_dendrogram_analysis(X_scaled, linkage_method="ward"):
    """Perform dendrogram analysis for agglomerative clustering"""
    # Calculate linkage matrix with selected method
    linkage_matrix = linkage(X_scaled, method=linkage_method)

    # Create dendrogram
    plt.figure(figsize=(12, 8))
    plt.clf()  # Clear the figure to avoid conflicts
    dendrogram(
        linkage_matrix, truncate_mode="level", p=10, show_leaf_counts=True
    )
    plt.xlabel("Samples or (number of clusters)")
    plt.ylabel("Distance")
    plt.title(
        f"Dendrogram for Hierarchical Clustering ({linkage_method.title()} linkage)"
    )
    plt.grid(True, alpha=0.3)

    # Convert to base64
    img_buffer = BytesIO()
    plt.savefig(img_buffer, format="png", dpi=150, bbox_inches="tight")
    img_buffer.seek(0)
    img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
    plt.close()

    return jsonify(
        {
            "image": img_base64,
            "linkage_matrix": linkage_matrix.tolist(),
            "linkage_method": linkage_method,
        }
    )


@app.route("/api/clustering_analysis", methods=["POST"])
def clustering_analysis():
    """Perform clustering analysis"""
    data = request.get_json()

    features = data.get("features", [])
    scaler_config = data.get("scaler_config", {})  # Per-feature scaler config
    algorithm = data.get("algorithm", "kmeans")
    params = data.get("params", {})

    # Get data
    with engine.connect() as conn:
        rows = (
            conn.execute(
                text(
                    """
            SELECT id, title, price_num, rooms, area_m2, price_per_m2, location, image_url, url
              FROM apartments
             WHERE area_m2 > 0
               AND price_per_m2 >= 1
               AND price_num > 0
               AND rooms > 0
               AND location IS NOT NULL
               AND location != ''
        """
                )
            )
            .mappings()
            .all()
        )

    df = pd.DataFrame(rows)
    # Set the database ID as the index to preserve it
    df.set_index("id", inplace=True)

    if not features or len(features) < 2:
        return jsonify({"error": "Select at least 2 features"})

    # Prepare data
    X = df[features].dropna()

    if len(X) < 2:
        return jsonify({"error": "Nedostatek dat pro clustering"})

    # Apply per-feature scaling
    X_scaled = X.copy()

    scalers = {
        "standard": StandardScaler(),
        "minmax": MinMaxScaler(),
        "robust": RobustScaler(),
        "power": PowerTransformer(method="yeo-johnson", standardize=True),
    }

    # Default scaler if no config provided
    default_scaler = "standard"

    for feature in features:
        scaler_type = scaler_config.get(feature, default_scaler)
        scaler = scalers.get(scaler_type, scalers[default_scaler])

        # Fit and transform each feature individually
        feature_data = X[[feature]].values
        try:
            scaled_feature = scaler.fit_transform(feature_data)
            X_scaled[feature] = scaled_feature.flatten()
        except Exception as e:
            # Fallback to standard scaler if transformation fails
            print(
                f"Warning: Failed to apply {scaler_type} to {feature}, using standard scaler: {e}"
            )
            scaler = scalers["standard"]
            scaled_feature = scaler.fit_transform(feature_data)
            X_scaled[feature] = scaled_feature.flatten()

    # Perform clustering
    labels = None
    if algorithm == "kmeans":
        n_clusters = params.get("n_clusters", 3)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)

    elif algorithm == "dbscan":
        eps = params.get("eps", 0.5)
        min_samples = params.get("min_samples", 5)
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(X_scaled)

    elif algorithm == "agglomerative":
        n_clusters = params.get("n_clusters", 3)
        linkage_method = params.get("linkage", "ward")
        agg = AgglomerativeClustering(
            n_clusters=n_clusters, linkage=linkage_method
        )
        labels = agg.fit_predict(X_scaled)

    # Add cluster labels to dataframe
    X_with_clusters = X.copy()
    X_with_clusters["cluster"] = labels

    # Add the original IDs back to the dataframe for mapping
    X_with_clusters["id"] = df.index.values

    # Calculate silhouette score
    if len(set(labels)) > 1:
        silhouette_avg = silhouette_score(X_scaled, labels)
    else:
        silhouette_avg = 0

    # PCA for visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # Create PCA plot
    plt.figure(figsize=(10, 8))
    plt.clf()  # Clear the figure to avoid conflicts
    unique_labels = set(labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

    for i, label in enumerate(unique_labels):
        if label == -1:  # Noise points for DBSCAN
            plt.scatter(
                X_pca[labels == label, 0],
                X_pca[labels == label, 1],
                c="black",
                marker="x",
                s=50,
                alpha=0.6,
                label="Noise",
            )
        else:
            plt.scatter(
                X_pca[labels == label, 0],
                X_pca[labels == label, 1],
                c=[colors[i]],
                label=f"Cluster {label}",
                alpha=0.7,
            )

    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)")
    plt.title("PCA Cluster Visualization")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Convert PCA plot to base64
    pca_buffer = BytesIO()
    plt.savefig(pca_buffer, format="png", dpi=150, bbox_inches="tight")
    pca_buffer.seek(0)
    pca_base64 = base64.b64encode(pca_buffer.getvalue()).decode()
    plt.close()

    # Create boxplots for EDA
    n_features = len(features)
    n_cols = min(2, n_features)
    n_rows = (n_features + 1) // 2

    plt.figure(figsize=(14, 5 * n_rows))
    plt.clf()  # Clear the figure to avoid conflicts
    axes = plt.subplots(n_rows, n_cols)[1]
    if n_features == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = [axes] if n_cols == 1 else axes
    else:
        axes = axes.flatten()

    feature_names = {
        "price_num": "Price (CZK)",
        "rooms": "Number of Rooms",
        "area_m2": "Size (m²)",
        "price_per_m2": "Price per m² (CZK)",
    }

    # Define colors for clusters
    colors = [
        "#FF6B6B",
        "#4ECDC4",
        "#45B7D1",
        "#96CEB4",
        "#FFEAA7",
        "#DDA0DD",
        "#98D8C8",
        "#F7DC6F",
    ]

    for i, feature in enumerate(features):
        if i < len(axes):
            cluster_data = []
            cluster_labels = []
            cluster_colors = []

            for j, label in enumerate(unique_labels):
                if label != -1:  # Skip noise
                    cluster_values = X_with_clusters[
                        X_with_clusters["cluster"] == label
                    ][feature]
                    cluster_data.append(cluster_values)
                    cluster_labels.append(f"Cluster {label}")
                    cluster_colors.append(colors[j % len(colors)])

            if cluster_data:
                # Create boxplot with custom styling
                bp = axes[i].boxplot(
                    cluster_data, tick_labels=cluster_labels, patch_artist=True
                )

                # Color the boxes
                for patch, color in zip(bp["boxes"], cluster_colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.6)
                    patch.set_edgecolor("black")
                    patch.set_linewidth(1.2)

                # Style the whiskers, caps, and medians
                for element in ["whiskers", "caps", "medians"]:
                    for item in bp[element]:
                        item.set_color("black")
                        item.set_linewidth(1.2)

                # Style outliers
                for item in bp["fliers"]:
                    item.set_marker("o")
                    item.set_markerfacecolor("red")
                    item.set_markeredgecolor("black")
                    item.set_markersize(3)
                    item.set_alpha(0.7)

                title = f"{feature_names.get(feature, feature)} by Clusters"
                axes[i].set_title(title, fontsize=13, fontweight="bold", pad=15)
                axes[i].set_ylabel(
                    feature_names.get(feature, feature),
                    fontsize=11,
                    fontweight="bold",
                )
                axes[i].set_xlabel("Cluster", fontsize=11, fontweight="bold")
                axes[i].tick_params(axis="x", rotation=45, labelsize=10)
                axes[i].tick_params(axis="y", labelsize=10)

                # Format y-axis to avoid scientific notation
                axes[i].yaxis.set_major_formatter(
                    plt.FuncFormatter(
                        lambda x, p: f"{x:,.0f}" if x >= 1000 else f"{x:.0f}"
                    )
                )

                # Add subtle grid
                axes[i].grid(True, alpha=0.2, linestyle="-", linewidth=0.5)

                # Set background color
                axes[i].set_facecolor("#fafafa")

    # Hide unused subplots
    for i in range(len(features), len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()

    # Convert boxplot to base64
    boxplot_buffer = BytesIO()
    plt.savefig(boxplot_buffer, format="png", dpi=150, bbox_inches="tight")
    boxplot_buffer.seek(0)
    boxplot_base64 = base64.b64encode(boxplot_buffer.getvalue()).decode()
    plt.close()

    # Cluster statistics
    cluster_stats = {}
    for label in unique_labels:
        if label != -1:
            cluster_data = X_with_clusters[X_with_clusters["cluster"] == label]
            stats = {}
            for feature in features:
                values = cluster_data[feature]
                stats[feature] = {
                    "count": len(values),
                    "mean": float(values.mean()),
                    "std": float(values.std()),
                    "min": float(values.min()),
                    "max": float(values.max()),
                }
            cluster_stats[f"cluster_{label}"] = stats

    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    n_noise = len(labels[labels == -1]) if -1 in unique_labels else 0

    # Prepare property data for each cluster by mapping to original listings
    property_data = {}

    for cluster_id in unique_labels:
        if cluster_id != -1:  # Skip noise
            cluster_properties = X_with_clusters[
                X_with_clusters["cluster"] == cluster_id
            ]

            # Get the IDs of properties in this cluster
            cluster_ids = cluster_properties["id"].tolist()

            # Fetch full property details from database
            with engine.connect() as conn:
                # Use SQLAlchemy's bindparam for proper parameter binding
                from sqlalchemy import bindparam

                query = text(
                    """
                    SELECT id, title, price_num, rooms, area_m2, price_per_m2, 
                           location, image_url, url
                    FROM apartments 
                    WHERE id IN :ids
                    """
                ).bindparams(bindparam("ids", expanding=True))

                full_properties = (
                    conn.execute(query, {"ids": cluster_ids}).mappings().all()
                )

            # Convert to list of dictionaries
            property_data[f"cluster_{cluster_id}"] = [
                dict(prop) for prop in full_properties
            ]

    return jsonify(
        {
            "pca_plot": pca_base64,
            "boxplot": boxplot_base64,
            "silhouette_score": float(silhouette_avg),
            "cluster_stats": cluster_stats,
            "property_data": property_data,
            "n_clusters": n_clusters,
            "n_noise": n_noise,
        }
    )


# Global state storage for K-means visualization
kmeans_state = {
    "data": None,
    "centroids": None,
    "labels": None,
    "iteration": 0,
    "k": 4,
    "n_samples": 100,
    "colors": [
        "red",
        "blue",
        "green",
        "orange",
        "purple",
        "brown",
        "pink",
        "gray",
    ],
}


@app.route("/api/kmeans_step", methods=["POST"])
def kmeans_step():
    """Generate K-means step visualization with persistent state"""
    global kmeans_state

    data = request.get_json()
    step = data.get("step", 0)  # 0: init, 1: assign, 2: move, etc.
    iteration = data.get("iteration", 0)
    k = data.get("k", 4)
    n_samples = data.get("n_samples", 100)

    # Initialize data if not exists or if parameters changed
    if (
        kmeans_state["data"] is None
        or kmeans_state["k"] != k
        or kmeans_state["n_samples"] != n_samples
    ):
        from sklearn.datasets import make_blobs

        kmeans_state["data"], _ = make_blobs(
            n_samples=n_samples,
            centers=4,
            cluster_std=1.5,
            random_state=42,
            center_box=(-10, 10),
        )
        kmeans_state["k"] = k
        kmeans_state["n_samples"] = n_samples
        kmeans_state["centroids"] = None
        kmeans_state["labels"] = None
        kmeans_state["iteration"] = 0
        kmeans_state["converged"] = False

    X = kmeans_state["data"]
    colors = kmeans_state["colors"]

    # Create visualization based on step
    plt.figure(figsize=(10, 8))
    plt.clf()  # Clear the figure to avoid conflicts

    if step == 0:  # Initial data
        plt.scatter(X[:, 0], X[:, 1], c="lightblue", alpha=0.6, s=50)
        plt.title("K-means: Initial Data Points")
        kmeans_state["centroids"] = None
        kmeans_state["labels"] = None
        kmeans_state["iteration"] = 0

    elif step == 1:  # Random centroids
        # Generate random centroids and store them
        min_x, max_x = X[:, 0].min(), X[:, 0].max()
        min_y, max_y = X[:, 1].min(), X[:, 1].max()
        kmeans_state["centroids"] = np.random.uniform(
            low=[min_x, min_y], high=[max_x, max_y], size=(k, 2)
        )

        plt.scatter(X[:, 0], X[:, 1], c="lightblue", alpha=0.6, s=50)
        for i, centroid in enumerate(kmeans_state["centroids"]):
            plt.scatter(
                centroid[0],
                centroid[1],
                c=colors[i % len(colors)],
                marker="x",
                s=200,
                linewidths=3,
                label=f"Centroid {i+1}",
            )
        plt.title("K-means: Random Initial Centroids")
        plt.legend()

    elif step == 2:  # Assign points to centroids
        if kmeans_state["centroids"] is None:
            # If no centroids exist, generate them first
            min_x, max_x = X[:, 0].min(), X[:, 0].max()
            min_y, max_y = X[:, 1].min(), X[:, 1].max()
            kmeans_state["centroids"] = np.random.uniform(
                low=[min_x, min_y], high=[max_x, max_y], size=(k, 2)
            )

        # Assign points to nearest centroid
        distances = np.sqrt(
            (
                (
                    X[:, np.newaxis, :]
                    - kmeans_state["centroids"][np.newaxis, :, :]
                )
                ** 2
            ).sum(axis=2)
        )
        kmeans_state["labels"] = np.argmin(distances, axis=1)

        for i in range(k):
            mask = kmeans_state["labels"] == i
            if np.any(mask):
                plt.scatter(
                    X[mask, 0],
                    X[mask, 1],
                    c=colors[i % len(colors)],
                    alpha=0.6,
                    s=50,
                    label=f"Cluster {i+1}",
                )

        # Plot centroids
        for i, centroid in enumerate(kmeans_state["centroids"]):
            plt.scatter(
                centroid[0],
                centroid[1],
                c=colors[i % len(colors)],
                marker="x",
                s=200,
                linewidths=3,
                label=f"Centroid {i+1}",
            )

        # Draw lines from points to centroids
        for i, point in enumerate(X):
            centroid = kmeans_state["centroids"][kmeans_state["labels"][i]]
            plt.plot(
                [point[0], centroid[0]],
                [point[1], centroid[1]],
                color=colors[kmeans_state["labels"][i] % len(colors)],
                alpha=0.3,
                linewidth=0.5,
            )

        plt.title(
            f"K-means: Assign Points to Nearest Centroid (Iteration {kmeans_state['iteration']})"
        )
        plt.legend()

    elif step == 3:  # Move centroids
        if kmeans_state["centroids"] is None or kmeans_state["labels"] is None:
            # If no centroids or labels exist, generate them first
            min_x, max_x = X[:, 0].min(), X[:, 0].max()
            min_y, max_y = X[:, 1].min(), X[:, 1].max()
            kmeans_state["centroids"] = np.random.uniform(
                low=[min_x, min_y], high=[max_x, max_y], size=(k, 2)
            )
            distances = np.sqrt(
                (
                    (
                        X[:, np.newaxis, :]
                        - kmeans_state["centroids"][np.newaxis, :, :]
                    )
                    ** 2
                ).sum(axis=2)
            )
            kmeans_state["labels"] = np.argmin(distances, axis=1)

        # Store old centroids
        old_centroids = kmeans_state["centroids"].copy()

        # Calculate new centroids
        new_centroids = np.array(
            [
                (
                    X[kmeans_state["labels"] == i].mean(axis=0)
                    if np.any(kmeans_state["labels"] == i)
                    else kmeans_state["centroids"][i]
                )  # Keep old centroid if cluster is empty
                for i in range(k)
            ]
        )

        # Check for convergence
        centroid_shift = np.sqrt(
            ((new_centroids - old_centroids) ** 2).sum(axis=1)
        ).max()
        converged = centroid_shift < 1e-6

        # Update state
        kmeans_state["centroids"] = new_centroids
        kmeans_state["iteration"] += 1
        kmeans_state["converged"] = converged

        # Plot clusters
        for i in range(k):
            mask = kmeans_state["labels"] == i
            if np.any(mask):
                plt.scatter(
                    X[mask, 0],
                    X[mask, 1],
                    c=colors[i % len(colors)],
                    alpha=0.6,
                    s=50,
                    label=f"Cluster {i+1}",
                )

        # Plot old centroids (faded)
        for i, centroid in enumerate(old_centroids):
            plt.scatter(
                centroid[0],
                centroid[1],
                c=colors[i % len(colors)],
                marker="x",
                s=200,
                linewidths=3,
                alpha=0.5,
            )

        # Plot new centroids
        for i, centroid in enumerate(new_centroids):
            plt.scatter(
                centroid[0],
                centroid[1],
                c=colors[i % len(colors)],
                marker="o",
                s=200,
                linewidths=3,
                edgecolors="black",
            )
            # Draw arrow from old to new centroid
            plt.annotate(
                "",
                xy=centroid,
                xytext=old_centroids[i],
                arrowprops=dict(
                    arrowstyle="->", color=colors[i % len(colors)], lw=2
                ),
            )

        plt.title(
            f"K-means: Move Centroids to Mean (Iteration {kmeans_state['iteration']})"
        )
        plt.legend()

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True, alpha=0.3)

    # Convert to base64
    img_buffer = BytesIO()
    plt.savefig(img_buffer, format="png", dpi=150, bbox_inches="tight")
    img_buffer.seek(0)
    img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
    plt.close()

    # Convert centroids to list, handling NaN values
    centroids_list = None
    if kmeans_state["centroids"] is not None:
        centroids_array = kmeans_state["centroids"]
        # Replace NaN values with None for JSON serialization
        centroids_list = []
        for centroid in centroids_array:
            centroid_list = []
            for coord in centroid:
                if np.isnan(coord):
                    centroid_list.append(None)
                else:
                    centroid_list.append(float(coord))
            centroids_list.append(centroid_list)

    return jsonify(
        {
            "image": img_base64,
            "iteration": kmeans_state["iteration"],
            "centroids": centroids_list,
            "converged": bool(kmeans_state.get("converged", False)),
        }
    )


@app.route("/api/kmeans_reset", methods=["POST"])
def kmeans_reset():
    """Reset K-means state"""
    global kmeans_state

    kmeans_state["centroids"] = None
    kmeans_state["labels"] = None
    kmeans_state["iteration"] = 0
    kmeans_state["converged"] = False

    return jsonify({"status": "reset"})


# Global state storage for DBSCAN visualization
dbscan_state = {
    "data": None,
    "eps": 0.5,
    "min_samples": 5,
    "n_samples": 100,
    "neighbors": None,
    "core_points": None,
    "border_points": None,
    "noise_points": None,
    "clusters": None,
    "colors": [
        "red",
        "blue",
        "green",
        "orange",
        "purple",
        "brown",
        "pink",
        "gray",
    ],
}


def find_neighbors(X, eps):
    """Find neighbors for each point within epsilon distance"""
    neighbors = {}
    for i, point in enumerate(X):
        neighbors[i] = []
        for j, other_point in enumerate(X):
            if i != j and np.linalg.norm(point - other_point) <= eps:
                neighbors[i].append(j)
    return neighbors


def classify_points(neighbors, min_samples):
    """Classify points as core, border, or noise"""
    core_points = []
    border_points = []
    noise_points = []

    for i, neighbor_list in neighbors.items():
        if len(neighbor_list) >= min_samples:
            core_points.append(i)
        elif len(neighbor_list) > 0:
            # Check if this point is neighbor of any core point
            is_border = False
            for neighbor in neighbor_list:
                if len(neighbors[neighbor]) >= min_samples:
                    is_border = True
                    break
            if is_border:
                border_points.append(i)
            else:
                noise_points.append(i)
        else:
            noise_points.append(i)

    return core_points, border_points, noise_points


def form_clusters(core_points, border_points, neighbors):
    """Form clusters by connecting core points"""
    clusters = {}
    cluster_id = 0
    visited = set()

    # Handle None values
    if core_points is None:
        core_points = []
    if border_points is None:
        border_points = []
    if neighbors is None:
        neighbors = {}

    for core_point in core_points:
        if core_point in visited:
            continue

        # Start new cluster
        cluster = []
        queue = [core_point]
        visited.add(core_point)

        while queue:
            current = queue.pop(0)
            cluster.append(current)

            # Add all neighbors that are core or border points
            for neighbor in neighbors[current]:
                if neighbor not in visited and (
                    neighbor in core_points or neighbor in border_points
                ):
                    visited.add(neighbor)
                    queue.append(neighbor)

        if cluster:
            clusters[cluster_id] = cluster
            cluster_id += 1

    return clusters


@app.route("/api/dbscan_step", methods=["POST"])
def dbscan_step():
    """Generate detailed DBSCAN step visualization with persistent state"""
    global dbscan_state

    data = request.get_json()
    step = data.get(
        "step", 0
    )  # 0: data, 1: epsilon circles, 2: neighbors, 3: core points, 4: border points, 5: clusters, 6: final result
    eps = data.get("eps", 0.5)
    min_samples = data.get("min_samples", 5)
    n_samples = data.get("n_samples", 100)

    # Initialize data if not exists or if parameters changed
    if (
        dbscan_state["data"] is None
        or dbscan_state["eps"] != eps
        or dbscan_state["min_samples"] != min_samples
        or dbscan_state["n_samples"] != n_samples
    ):
        from sklearn.datasets import make_circles

        dbscan_state["data"], _ = make_circles(
            n_samples=n_samples, noise=0.1, factor=0.3, random_state=42
        )
        dbscan_state["data"] *= 5  # Scale up for better visualization
        dbscan_state["eps"] = eps
        dbscan_state["min_samples"] = min_samples
        dbscan_state["n_samples"] = n_samples
        dbscan_state["neighbors"] = None
        dbscan_state["core_points"] = None
        dbscan_state["border_points"] = None
        dbscan_state["noise_points"] = None
        dbscan_state["clusters"] = None

    X = dbscan_state["data"]
    colors = dbscan_state["colors"]

    # Create visualization based on step
    plt.figure(figsize=(12, 10))
    plt.clf()  # Clear the figure to avoid conflicts

    if step == 0:  # Initial data
        plt.scatter(X[:, 0], X[:, 1], c="lightblue", alpha=0.6, s=50)
        plt.title("DBSCAN: Initial Data Points")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.grid(True, alpha=0.3)

    elif step == 1:  # Epsilon circles
        plt.scatter(X[:, 0], X[:, 1], c="lightblue", alpha=0.6, s=50)

        # Draw epsilon circles around each point
        for point in X:
            circle = patches.Circle(
                point, eps, fill=False, color="gray", alpha=0.3, linestyle="--"
            )
            plt.gca().add_patch(circle)

        plt.title(f"DBSCAN: Epsilon Neighborhoods (ε = {eps})")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.grid(True, alpha=0.3)

    elif step == 2:  # Show neighbor connections
        # Calculate neighbors if not done yet
        if dbscan_state["neighbors"] is None:
            dbscan_state["neighbors"] = find_neighbors(X, eps)

        plt.scatter(X[:, 0], X[:, 1], c="lightblue", alpha=0.6, s=50)

        # Draw epsilon circles
        for point in X:
            circle = patches.Circle(
                point, eps, fill=False, color="gray", alpha=0.2, linestyle="--"
            )
            plt.gca().add_patch(circle)

        # Draw connections to neighbors
        for i, point in enumerate(X):
            for neighbor_idx in dbscan_state["neighbors"][i]:
                neighbor_point = X[neighbor_idx]
                plt.plot(
                    [point[0], neighbor_point[0]],
                    [point[1], neighbor_point[1]],
                    color="red",
                    alpha=0.4,
                    linewidth=0.5,
                )

        plt.title(f"DBSCAN: Neighbor Connections (ε = {eps})")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.grid(True, alpha=0.3)

    elif step == 3:  # Identify core points
        # Calculate neighbors and classify points if not done yet
        if dbscan_state["neighbors"] is None:
            dbscan_state["neighbors"] = find_neighbors(X, eps)
        if dbscan_state["core_points"] is None:
            core_points, border_points, noise_points = classify_points(
                dbscan_state["neighbors"], min_samples
            )
            dbscan_state["core_points"] = core_points
            dbscan_state["border_points"] = border_points
            dbscan_state["noise_points"] = noise_points

        plt.scatter(X[:, 0], X[:, 1], c="lightgray", alpha=0.3, s=30)

        # Draw epsilon circles
        for point in X:
            circle = patches.Circle(
                point, eps, fill=False, color="gray", alpha=0.1, linestyle="--"
            )
            plt.gca().add_patch(circle)

        # Highlight core points
        core_points = dbscan_state["core_points"]
        if core_points is not None and len(core_points) > 0:
            plt.scatter(
                X[core_points, 0],
                X[core_points, 1],
                c="red",
                marker="o",
                s=100,
                alpha=0.8,
                label=f"Core Points ({len(core_points)})",
                edgecolors="black",
            )

        plt.title(f"DBSCAN: Core Points (≥{min_samples} neighbors)")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.legend()
        plt.grid(True, alpha=0.3)

    elif step == 4:  # Identify border points
        plt.scatter(X[:, 0], X[:, 1], c="lightgray", alpha=0.3, s=30)

        # Draw epsilon circles
        for point in X:
            circle = patches.Circle(
                point, eps, fill=False, color="gray", alpha=0.1, linestyle="--"
            )
            plt.gca().add_patch(circle)

        # Highlight core points
        core_points = dbscan_state["core_points"]
        if core_points is not None and len(core_points) > 0:
            plt.scatter(
                X[core_points, 0],
                X[core_points, 1],
                c="red",
                marker="o",
                s=100,
                alpha=0.8,
                label=f"Core Points ({len(core_points)})",
                edgecolors="black",
            )

        # Highlight border points
        border_points = dbscan_state["border_points"]
        if border_points is not None and len(border_points) > 0:
            plt.scatter(
                X[border_points, 0],
                X[border_points, 1],
                c="blue",
                marker="s",
                s=100,
                alpha=0.8,
                label=f"Border Points ({len(border_points)})",
                edgecolors="black",
            )

        plt.title(f"DBSCAN: Border Points (neighbors of core points)")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.legend()
        plt.grid(True, alpha=0.3)

    elif step == 5:  # Identify noise points
        plt.scatter(X[:, 0], X[:, 1], c="lightgray", alpha=0.3, s=30)

        # Draw epsilon circles
        for point in X:
            circle = patches.Circle(
                point, eps, fill=False, color="gray", alpha=0.1, linestyle="--"
            )
            plt.gca().add_patch(circle)

        # Highlight core points
        core_points = dbscan_state["core_points"]
        if core_points is not None and len(core_points) > 0:
            plt.scatter(
                X[core_points, 0],
                X[core_points, 1],
                c="red",
                marker="o",
                s=100,
                alpha=0.8,
                label=f"Core Points ({len(core_points)})",
                edgecolors="black",
            )

        # Highlight border points
        border_points = dbscan_state["border_points"]
        if border_points is not None and len(border_points) > 0:
            plt.scatter(
                X[border_points, 0],
                X[border_points, 1],
                c="blue",
                marker="s",
                s=100,
                alpha=0.8,
                label=f"Border Points ({len(border_points)})",
                edgecolors="black",
            )

        # Highlight noise points
        noise_points = dbscan_state["noise_points"]
        if noise_points is not None and len(noise_points) > 0:
            plt.scatter(
                X[noise_points, 0],
                X[noise_points, 1],
                c="black",
                marker="x",
                s=100,
                alpha=0.8,
                label=f"Noise Points ({len(noise_points)})",
            )

        plt.title(f"DBSCAN: Noise Points (isolated or sparse)")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.legend()
        plt.grid(True, alpha=0.3)

    elif step == 6:  # Form clusters
        # Form clusters if not done yet
        if dbscan_state["clusters"] is None:
            dbscan_state["clusters"] = form_clusters(
                dbscan_state["core_points"],
                dbscan_state["border_points"],
                dbscan_state["neighbors"],
            )

        plt.scatter(X[:, 0], X[:, 1], c="lightgray", alpha=0.3, s=30)

        # Draw epsilon circles
        for point in X:
            circle = patches.Circle(
                point, eps, fill=False, color="gray", alpha=0.1, linestyle="--"
            )
            plt.gca().add_patch(circle)

        # Color clusters
        clusters = dbscan_state["clusters"]
        for cluster_id, cluster_points in clusters.items():
            plt.scatter(
                X[cluster_points, 0],
                X[cluster_points, 1],
                c=colors[cluster_id % len(colors)],
                s=80,
                alpha=0.8,
                label=f"Cluster {cluster_id + 1} ({len(cluster_points)} points)",
                edgecolors="black",
            )

        # Highlight noise points
        noise_points = dbscan_state["noise_points"]
        if noise_points is not None and len(noise_points) > 0:
            plt.scatter(
                X[noise_points, 0],
                X[noise_points, 1],
                c="black",
                marker="x",
                s=100,
                alpha=0.8,
                label=f"Noise Points ({len(noise_points)})",
            )

        plt.title(
            f"DBSCAN: Formed Clusters (ε={eps}, min_samples={min_samples})"
        )
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.legend()
        plt.grid(True, alpha=0.3)

    elif step == 7:  # Final result with statistics
        # Use sklearn DBSCAN for final result
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(X)

        plt.scatter(X[:, 0], X[:, 1], c="lightgray", alpha=0.3, s=30)

        # Draw epsilon circles
        for point in X:
            circle = patches.Circle(
                point, eps, fill=False, color="gray", alpha=0.1, linestyle="--"
            )
            plt.gca().add_patch(circle)

        # Color points based on cluster assignment
        unique_labels = set(labels)
        for k in unique_labels:
            if k == -1:  # Noise points
                mask = labels == k
                plt.scatter(
                    X[mask, 0],
                    X[mask, 1],
                    c="black",
                    marker="x",
                    s=80,
                    alpha=0.8,
                    label=f"Noise ({np.sum(mask)} points)",
                )
            else:
                mask = labels == k
                plt.scatter(
                    X[mask, 0],
                    X[mask, 1],
                    c=colors[k % len(colors)],
                    s=80,
                    alpha=0.8,
                    label=f"Cluster {k+1} ({np.sum(mask)} points)",
                )

        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        n_noise = list(labels).count(-1)

        plt.title(f"DBSCAN: Final Clustering Result")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Add detailed statistics text
        stats_text = f"""Final Statistics:
Clusters: {n_clusters}
Noise: {n_noise}
Total Points: {len(X)}
ε = {eps}, min_samples = {min_samples}"""
        plt.text(
            0.02,
            0.98,
            stats_text,
            transform=plt.gca().transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat"),
            fontsize=10,
        )

    plt.tight_layout()

    # Convert to base64
    img_buffer = BytesIO()
    plt.savefig(img_buffer, format="png", dpi=150, bbox_inches="tight")
    img_buffer.seek(0)
    img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
    plt.close()

    return jsonify(
        {
            "image": img_base64,
            "step": step,
            "eps": eps,
            "min_samples": min_samples,
        }
    )


@app.route("/api/dbscan_reset", methods=["POST"])
def dbscan_reset():
    """Reset DBSCAN state"""
    global dbscan_state

    dbscan_state["neighbors"] = None
    dbscan_state["core_points"] = None
    dbscan_state["border_points"] = None
    dbscan_state["noise_points"] = None
    dbscan_state["clusters"] = None

    return jsonify({"status": "reset"})


@app.route("/api/kmeans_animate", methods=["POST"])
def kmeans_animate():
    """Run K-means animation automatically"""
    global kmeans_state

    data = request.get_json()
    k = data.get("k", 4)
    n_samples = data.get("n_samples", 100)
    max_iterations = data.get("max_iterations", 20)

    # Initialize data
    if (
        kmeans_state["data"] is None
        or kmeans_state["k"] != k
        or kmeans_state["n_samples"] != n_samples
    ):
        from sklearn.datasets import make_blobs

        kmeans_state["data"], _ = make_blobs(
            n_samples=n_samples,
            centers=4,
            cluster_std=1.5,
            random_state=42,
            center_box=(-10, 10),
        )
        kmeans_state["k"] = k
        kmeans_state["n_samples"] = n_samples
        kmeans_state["centroids"] = None
        kmeans_state["labels"] = None
        kmeans_state["iteration"] = 0

    # Reset state for animation
    kmeans_state["centroids"] = None
    kmeans_state["labels"] = None
    kmeans_state["iteration"] = 0

    # Generate initial centroids
    X = kmeans_state["data"]
    min_x, max_x = X[:, 0].min(), X[:, 0].max()
    min_y, max_y = X[:, 1].min(), X[:, 1].max()
    kmeans_state["centroids"] = np.random.uniform(
        low=[min_x, min_y], high=[max_x, max_y], size=(k, 2)
    )

    # Run iterations
    for iteration in range(max_iterations):
        # Assign points to nearest centroid
        distances = np.sqrt(
            (
                (
                    X[:, np.newaxis, :]
                    - kmeans_state["centroids"][np.newaxis, :, :]
                )
                ** 2
            ).sum(axis=2)
        )
        kmeans_state["labels"] = np.argmin(distances, axis=1)

        # Calculate new centroids
        new_centroids = np.array(
            [
                (
                    X[kmeans_state["labels"] == i].mean(axis=0)
                    if np.any(kmeans_state["labels"] == i)
                    else kmeans_state["centroids"][i]
                )  # Keep old centroid if cluster is empty
                for i in range(k)
            ]
        )

        # Check for convergence (more strict tolerance)
        if np.allclose(kmeans_state["centroids"], new_centroids, atol=1e-6):
            break

        kmeans_state["centroids"] = new_centroids
        kmeans_state["iteration"] = iteration + 1

    return jsonify(
        {
            "status": "completed",
            "iterations": kmeans_state["iteration"],
            "converged": iteration < max_iterations - 1,
        }
    )


@app.route("/api/dbscan_animate", methods=["POST"])
def dbscan_animate():
    """Run DBSCAN animation automatically"""
    global dbscan_state

    data = request.get_json()
    eps = data.get("eps", 0.5)
    min_samples = data.get("min_samples", 5)
    n_samples = data.get("n_samples", 100)

    # Initialize data
    if (
        dbscan_state["data"] is None
        or dbscan_state["eps"] != eps
        or dbscan_state["min_samples"] != min_samples
        or dbscan_state["n_samples"] != n_samples
    ):
        from sklearn.datasets import make_circles

        dbscan_state["data"], _ = make_circles(
            n_samples=n_samples, noise=0.1, factor=0.3, random_state=42
        )
        dbscan_state["data"] *= 5  # Scale up for better visualization
        dbscan_state["eps"] = eps
        dbscan_state["min_samples"] = min_samples
        dbscan_state["n_samples"] = n_samples
        dbscan_state["neighbors"] = None
        dbscan_state["core_points"] = None
        dbscan_state["border_points"] = None
        dbscan_state["noise_points"] = None
        dbscan_state["clusters"] = None

    # Run DBSCAN steps
    X = dbscan_state["data"]

    # Step 1: Find neighbors
    dbscan_state["neighbors"] = find_neighbors(X, eps)

    # Step 2: Classify points
    core_points, border_points, noise_points = classify_points(
        dbscan_state["neighbors"], min_samples
    )
    dbscan_state["core_points"] = core_points
    dbscan_state["border_points"] = border_points
    dbscan_state["noise_points"] = noise_points

    # Step 3: Form clusters
    dbscan_state["clusters"] = form_clusters(
        core_points, border_points, dbscan_state["neighbors"]
    )

    return jsonify(
        {
            "status": "completed",
            "core_points": len(core_points) if core_points is not None else 0,
            "border_points": (
                len(border_points) if border_points is not None else 0
            ),
            "noise_points": (
                len(noise_points) if noise_points is not None else 0
            ),
            "clusters": (
                len(dbscan_state["clusters"])
                if dbscan_state["clusters"] is not None
                else 0
            ),
        }
    )


if __name__ == "__main__":
    app.run(debug=True)
