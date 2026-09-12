# Clustering Analysis Feature

This document describes the interactive clustering analysis feature added to the apartment listing application.

## Overview

`python app.py`

The clustering feature allows you to perform unsupervised machine learning analysis on apartment data using different clustering algorithms with customizable preprocessing options.

## Features

### 1. Feature Selection
- **Available Features:**
  - `price_num`: Total apartment price (Kč)
  - `rooms`: Number of rooms
  - `area_m2`: Apartment size in square meters
  - `price_per_m2`: Price per square meter (Kč)

- **Interactive Selection:** Click on feature cards to select/deselect features for clustering
- **Minimum Requirements:** At least 2 features must be selected

### 2. Data Visualization
- **Histograms:** View distribution of selected features before clustering
- **Interactive Display:** Histograms are generated dynamically based on selected features

### 3. Per-Feature Preprocessing
- **Scaler Options:**
  - **StandardScaler:** Standardization (mean=0, std=1)
  - **MinMaxScaler:** Scaling to range [0,1]
  - **RobustScaler:** Robust to outliers
  - **PowerTransformer:** Yeo-Johnson transformation for non-normal distributions

- **Individual Control:** Each selected feature can have its own scaler
- **Default:** StandardScaler is used if no scaler is specified

### 4. Cluster Number Analysis

#### Silhouette Analysis (K-means)
- **Purpose:** Determine optimal number of clusters for K-means
- **Method:** Calculates silhouette scores for different k values (2-10)
- **Output:** Plot showing silhouette scores vs. number of clusters
- **Recommendation:** Automatically suggests best k based on highest score
- **Interpretation:** Higher scores indicate better cluster separation

#### Elbow Method (K-means)
- **Purpose:** Find optimal number of clusters using inertia reduction
- **Method:** Calculates within-cluster sum of squares for different k values
- **Output:** Plot showing inertia vs. number of clusters
- **Recommendation:** Automatically detects elbow point using second derivative
- **Interpretation:** Look for the "elbow" where inertia reduction slows down

#### Dendrogram Analysis (Agglomerative)
- **Purpose:** Visualize hierarchical clustering structure
- **Method:** Creates dendrogram showing cluster merging process using selected linkage method
- **Output:** Hierarchical tree diagram with distance information
- **Interpretation:** Choose cluster count by "cutting" the dendrogram at desired level
- **Features:** 
  - Shows sample counts and distances between clusters
  - Uses selected linkage method (Ward, Complete, Average, Single)
  - Dynamic title shows current linkage method
  - Linkage method description in results

### 5. Clustering Algorithms

#### K-means
- **Parameters:**
  - `n_clusters`: Number of clusters (2-10) - can be optimized using silhouette/elbow analysis
- **Use Case:** Spherical clusters, known number of clusters
- **Analysis Tools:** Silhouette analysis, Elbow method

#### DBSCAN
- **Parameters:**
  - `eps`: Maximum distance between samples (0.1-2.0)
  - `min_samples`: Minimum samples in a neighborhood (2-20)
- **Use Case:** Arbitrary shaped clusters, outlier detection
- **Analysis Tools:** Parameter tuning based on data density

#### Agglomerative Clustering
- **Parameters:**
  - `n_clusters`: Number of clusters (2-10) - can be optimized using dendrogram analysis
  - `linkage`: Linkage criterion (ward, complete, average, single)
- **Use Case:** Hierarchical clustering, dendrogram analysis
- **Analysis Tools:** Dendrogram visualization

### 6. Visualization and Analysis

#### PCA Visualization
- **2D Projection:** Clusters visualized in 2D using Principal Component Analysis
- **Variance Explained:** Shows percentage of variance explained by each principal component
- **Color Coding:** Different colors for each cluster, black X for noise points (DBSCAN)

#### EDA Boxplots
- **Feature Distribution:** Boxplots showing distribution of each feature by cluster
- **Statistical Insights:** Compare feature values across different clusters
- **Outlier Detection:** Visual identification of outliers within clusters

#### Cluster Statistics
- **Metrics:** Count, mean, standard deviation, min, max for each feature in each cluster
- **Silhouette Score:** Overall clustering quality metric (-1 to 1, higher is better)
- **Noise Points:** Count of noise points (DBSCAN only)

## Usage Instructions

### 1. Access the Feature
- Navigate to `/clustering` from the main application
- Or click the "Clustering" button from the main page or analytics page

### 2. Select Features
- Click on feature cards to select features for analysis
- At least 2 features must be selected
- Click "Zobrazit histogramy vybraných vlastností" to view feature distributions

### 3. Configure Preprocessing
- For each selected feature, choose a scaler method
- Different features can use different scalers
- PowerTransformer is recommended for skewed distributions

### 4. Choose Algorithm and Parameters
- Select clustering algorithm (K-means, DBSCAN, or Agglomerative)
- Adjust algorithm-specific parameters
- Parameters are validated automatically

### 5. Optimize Cluster Count (Optional but Recommended)
- **For K-means:** Click "Silhouette" or "Elbow" buttons to analyze optimal cluster count
- **For Agglomerative:** 
  - Select linkage method (Ward, Complete, Average, Single)
  - Click "Dendrogram" button to visualize hierarchical structure
  - Dendrogram will use the selected linkage method
- Review analysis results and recommendations
- Adjust cluster count based on analysis findings

### 6. Run Analysis
- Click "Spustit clustering analýzu"
- Wait for processing to complete
- View results in the results section

## Technical Implementation

### Backend (Flask)
- **Routes:**
  - `/clustering`: Main clustering page
  - `/api/clustering_data`: Get apartment data for clustering
  - `/api/clustering_histogram`: Generate feature histograms
  - `/api/cluster_analysis`: Perform silhouette/elbow/dendrogram analysis
  - `/api/clustering_analysis`: Perform clustering analysis

- **Libraries Used:**
  - `scikit-learn`: Clustering algorithms and preprocessing
  - `pandas`: Data manipulation
  - `numpy`: Numerical operations
  - `matplotlib`: Plot generation
  - `seaborn`: Statistical visualization
  - `scipy`: Hierarchical clustering and dendrograms

### Frontend (HTML/JavaScript)
- **Interactive UI:** Bootstrap-based responsive design
- **Dynamic Configuration:** JavaScript-generated scaler options
- **Real-time Updates:** Immediate feedback on selections
- **Error Handling:** User-friendly error messages

## Example Use Cases

### 1. Price Segmentation
- **Features:** `price_num`, `area_m2`, `price_per_m2`
- **Scalers:** PowerTransformer for prices, StandardScaler for area
- **Algorithm:** K-means with 3-5 clusters
- **Insight:** Identify different price segments in the market

### 2. Size-Based Clustering
- **Features:** `rooms`, `area_m2`
- **Scalers:** MinMaxScaler for both features
- **Algorithm:** DBSCAN for irregular cluster shapes
- **Insight:** Find natural groupings by apartment size

### 3. Comprehensive Analysis
- **Features:** All available features
- **Scalers:** Mixed approach (PowerTransformer for prices, StandardScaler for others)
- **Algorithm:** Agglomerative clustering
- **Insight:** Holistic view of apartment market segments

## Dependencies

Required Python packages:
```
scrapy
flask
sqlalchemy
jinja2
scikit-learn
pandas
numpy
matplotlib
seaborn
plotly
scipy
```
