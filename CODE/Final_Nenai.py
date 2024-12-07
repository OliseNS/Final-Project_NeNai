#%% MODULE BEGINS
module_name = 'final_project'

'''
Version: 5.3

Description:
    Feature Generation for EEG Data with Additional clustering and evaluation matrix

Authors:
    NeNai: Olisemeka Nmarkwe and Sujana Mehta. (W0762669 and W0757459 respectively)

Date Created     :  12/03/2024
Date Last Updated:  12/06/2024

Doc:
    This module generates features from EEG data while addressing redundancy and enhancing modularity.

Notes:
    Ensure the data directory structure and channel names are correctly configured before running.
'''

#%% IMPORTS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if __name__ == "__main__":
    import os
import pickle as pckl
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew
from scipy.signal import butter, filtfilt, iirnotch
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, confusion_matrix
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from itertools import combinations
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from scipy.stats import zscore
from sklearn.impute import SimpleImputer


#%% USER INTERFACE              ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Base directory containing 'sb1' and 'sb2' folders

base_dir = os.path.join('CODE','INPUT')




#%% CONSTANTS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
DEFAULT_WINDOW_SIZE = 100  # Number of samples per window
DEFAULT_OVERLAP = 0.5      # 50% overlap between windows
DATA_FILES = [
    "TestData_CPz.csv",
    "TestData_M1.csv",
    "TestData_M2.csv",
    "TrainValidateData_CPz.csv",
    "TrainValidateData_M1.csv",
    "TrainValidateData_M2.csv"
]


#%% FUNCTION DEFINITIONS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# feature generations  --------------------------------------------------

def load_files_insession(subject, session, channel_name):
    pathSoi = rf"{base_dir}\{subject}\{session}"
    filelist = os.listdir(pathSoi)
    session_data = {}

    for soi_file in filelist:
        with open(f'{pathSoi}\\{soi_file}', 'rb') as fp:
            soi = pckl.load(fp)
        
        channel_info = soi['info']['eeg_info']['channels']
        channel_indices = [i for i, ch in enumerate(channel_info) if ch['label'][0] == channel_name]
        
        if channel_indices:
            index = channel_indices[0]
            session_data[soi_file] = soi['series'][index]
        
        sfreq = soi['info']['eeg_info']['effective_srate']

    return session_data, sfreq

def apply_notch_filter(data, fs, freqs):
    for freq in freqs:
        b, a = iirnotch(w0=freq, Q=30, fs=fs)
        data = filtfilt(b, a, data)
    return data

def apply_impedance_filter(data, fs, center, tolerance):
    low, high = center - tolerance, center + tolerance
    b, a = butter(N=2, Wn=[low, high], btype='bandstop', fs=fs)
    return filtfilt(b, a, data)

def apply_band_pass_filter(data, fs, lowcut, highcut):
    b, a = butter(N=2, Wn=[lowcut, highcut], btype='bandpass', fs=fs)
    return filtfilt(b, a, data)

def apply_rereferencing(data):
    reference = np.mean(data, axis=0)
    return data - reference

def get_stat_for_window(window_signal, se_val, sb_val, stream_id, index):
    mean = np.mean(window_signal)
    std_dev = np.std(window_signal)
    kur = kurtosis(window_signal)
    skewness = skew(window_signal)
    return {
        'sb': sb_val,
        'se': se_val,
        'stream_id': stream_id,
        'window_id': index,
        'mean': mean,
        'std': std_dev,
        'kur': kur,
        'skew': skewness           
    }

def get_features(df):
    grouped_df = df.groupby(['sb', 'se', 'stream_id'])[['mean', 'std', 'kur', 'skew']].agg([np.mean, np.std])
    result_df = grouped_df.assign(
        a1=lambda x: x['mean']['mean'],
        a2=lambda x: x['mean']['std'],
        a3=lambda x: x['std']['mean'],
        a4=lambda x: x['std']['std'],
        a5=lambda x: x['kur']['mean'],
        a6=lambda x: x['kur']['std'],
        a7=lambda x: x['skew']['mean'],
        a8=lambda x: x['skew']['std'],
    )


    return result_df.reset_index()

def create_target_class(sb_value):
    return 'Sb1' if sb_value.lower() == 'sb1' else 'Sb2'


def new_df(df):
    new_df = df[['a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8', 'target']].copy()
    new_df['sample_id'] = range(1, len(new_df) + 1)
    return new_df[['sample_id', 'target', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8']].sort_values(by='sample_id')


def create_datasets(subject1, session1, subject2, session2, channel):
    # Helper function to collect features for a given subject and session
    def collect_features(subject, session, channel):
        session_data, sfreq = load_files_insession(subject, session, channel)
        statistics = []

        for file_name, data in session_data.items():
            # Apply preprocessing steps
            data = apply_notch_filter(data, sfreq, [60, 120, 180, 240])
            data = apply_impedance_filter(data, sfreq, 125, 1)
            data = apply_band_pass_filter(data, sfreq, 0.5, 32)
            data = apply_rereferencing(data)

            # Apply windowing and feature extraction
            for i in range(0, len(data) - DEFAULT_WINDOW_SIZE + 1, int(DEFAULT_WINDOW_SIZE * (1 - DEFAULT_OVERLAP))):
                window = data[i:i + DEFAULT_WINDOW_SIZE]
                statistics.append(get_stat_for_window(window, session, subject, file_name, i))

        return pd.DataFrame(statistics)

    # Collect features for training/validation and testing
    train_validate_df = pd.concat([
        collect_features(subject1, session1, channel),
        collect_features(subject2, session1, channel)
    ])
    test_df = pd.concat([
        collect_features(subject1, session2, channel),
        collect_features(subject2, session2, channel)
    ])

    # Aggregate features and add target labels
    train_validate_df = get_features(train_validate_df)
    train_validate_df['target'] = train_validate_df['sb'].apply(create_target_class)

    test_df = get_features(test_df)
    test_df['target'] = test_df['sb'].apply(create_target_class)

    # Structure the final dataframes
    train_validate_df = new_df(train_validate_df)
    test_df = new_df(test_df)

    return train_validate_df, test_df

def save_and_plot(train_validate, test, channel_name):
    # Define the output directory
    output_dir = os.path.join('output', 'feature')
    os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist

    # Remove any empty rows in the DataFrames
    train_validate = train_validate.dropna().reset_index(drop=True)
    test = test.dropna().reset_index(drop=True)

    # Save datasets to CSV files
    train_validate_path = os.path.join(output_dir, f'TrainValidateData_{channel_name}.csv')
    test_path = os.path.join(output_dir, f'TestData_{channel_name}.csv')
    train_validate.to_csv(train_validate_path, index=False)
    test.to_csv(test_path, index=False)

    print(f"Saved train/validate data to {train_validate_path}")
    print(f"Saved test data to {test_path}")

    # Combine datasets for visualization
    features_df = pd.concat([train_validate, test])

    # Plot for Training Data
    plot_data(train_validate, channel_name, "Training Data")

    # Plot for Testing Data
    plot_data(test, channel_name, "Testing Data")


def plot_data(df, channel_name, dataset_type):
    # Define the output directory for plots
    plot_dir = os.path.join('output', 'feature')
    os.makedirs(plot_dir, exist_ok=True)

    continuous_features = df.select_dtypes(include=[np.number]).columns.drop(['sample_id']).tolist()

    print(f"Continuous Features for Plotting: {continuous_features}")

    # Histogram for categorical attribute
    plt.figure(figsize=(6, 4))
    sns.histplot(df['target'], kde=False, palette="Set2", discrete=True)
    plt.title(f"{dataset_type} - Categorical Distribution for Channel: {channel_name}", fontsize=14)
    plt.xlabel('Target Class (Sb1 / Sb2)', fontsize=10)
    plt.ylabel('Frequency', fontsize=10)
    plt.xticks([0, 1], ['Sb1', 'Sb2'])
    plt.tight_layout()
    hist_path = os.path.join(plot_dir, f"{dataset_type}_Categorical_Distribution_{channel_name}.png")
    plt.savefig(hist_path)
    print(f"Saved histogram plot to {hist_path}")
    plt.close()

    # Sorted Bar Charts for Continuous Attributes
    num_cols_sorted = 2
    num_rows_sorted = (len(continuous_features) - 1) // num_cols_sorted + 1

    plt.figure(figsize=(7 * num_cols_sorted, 4 * num_rows_sorted))
    for i, column in enumerate(continuous_features, start=1):
        plt.subplot(num_rows_sorted, num_cols_sorted, i)
        sorted_values = df[column].sort_values().values
        plt.bar(range(len(sorted_values)), sorted_values, color='skyblue')
        plt.title(f"Sorted Distribution: {column}", fontsize=10, pad=5)
        plt.xlabel("Sorted Index", fontsize=8)
        plt.ylabel("Value", fontsize=8)
        plt.grid(axis='y', linestyle='--', alpha=0.6)
        plt.xticks(fontsize=7)
        plt.yticks(fontsize=7)
    plt.tight_layout(pad=2.0, w_pad=1.5, h_pad=2.5)
    plt.suptitle(f"{dataset_type} - Sorted Distributions for Channel: {channel_name}", fontsize=16, y=1.02)
    sorted_bar_path = os.path.join(plot_dir, f"{dataset_type}_Sorted_Distributions_{channel_name}.png")
    plt.savefig(sorted_bar_path, bbox_inches="tight")
    print(f"Saved sorted bar chart to {sorted_bar_path}")
    plt.close()


# feature generations ends   --------------------------------------------------
def standarize_data(df, feature_columns, apply_pca=True):
    """
    Preprocess the data by removing low-variance features, handling outliers,
    imputing missing values, normalizing features, and applying PCA if needed.

    Args:
        df (DataFrame): The input data, including the target column.
        feature_columns (list): List of feature columns to preprocess.
        apply_pca (bool): Whether to conditionally apply PCA.

    Returns:
        DataFrame: The preprocessed data, including the target column.
    """
    # Retain the target column
    if "target" in df.columns:
        target_col = df["target"]
    else:
        target_col = None

    # 1. Remove low-variance features
    print("Applying Variance Thresholding...")
    selector = VarianceThreshold(threshold=0.01)
    filtered_features = selector.fit_transform(df[feature_columns])
    filtered_columns = [feature_columns[i] for i in selector.get_support(indices=True)]
    df_filtered = pd.DataFrame(filtered_features, columns=filtered_columns)

    # 2. Handle outliers
    print("Handling outliers...")
    for col in df_filtered.columns:
        lower = df_filtered[col].quantile(0.05)
        upper = df_filtered[col].quantile(0.95)
        df_filtered[col] = np.clip(df_filtered[col], lower, upper)

    # 3. Impute missing values
    print("Imputing missing values...")
    imputer = SimpleImputer(strategy="mean")
    imputed_features = imputer.fit_transform(df_filtered)
    df_imputed = pd.DataFrame(imputed_features, columns=df_filtered.columns)

    # 4. Normalize features
    print("Normalizing features...")
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df_imputed)
    df_scaled = pd.DataFrame(scaled_features, columns=df_imputed.columns)

    # 5. Conditionally apply PCA
    if apply_pca:
        print("Evaluating if PCA is needed...")
        total_variance = np.sum(np.var(df_scaled, axis=0))
        print(f"Total variance in data: {total_variance}")

        if total_variance > 0.01:
            print("Applying PCA...")
            pca = PCA(n_components=3)
            pca_features = pca.fit_transform(df_scaled)
            df_pca = pd.DataFrame(pca_features, columns=['PCA1', 'PCA2', 'PCA3'])

            # Add target column back if it exists
            if target_col is not None:
                df_pca["target"] = target_col.values

            return df_pca

    # Add target column back to scaled data if PCA is not applied
    if target_col is not None:
        df_scaled["target"] = target_col.values

    return df_scaled



# Clustering  --------------------------------------------------

def load_data(files, base_dir="output/feature"):
    """Load CSV files into a dictionary of DataFrames from the specified directory."""
    data_frames = {}
    for file in files:
        file_path = os.path.join(base_dir, file)
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            data_frames[file] = df
        else:
            print(f"File not found: {file_path}")
    return data_frames

def preprocess_data(df):
    """Preprocess data by dropping rows with missing values."""
    df_preprocessed = df.dropna()
    return df_preprocessed

def calculate_metrics(df, feature_columns, target_column, n_clusters=2):
    """Calculate clustering and evaluation metrics."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['cluster'] = kmeans.fit_predict(df[feature_columns])
    centroids = kmeans.cluster_centers_

    # Cohesion and Separation
    cohesion = kmeans.inertia_
    separation = sum(
        sum((centroids[i] - centroids[j]) ** 2)
        for i in range(n_clusters) for j in range(i + 1, n_clusters)
    )

    # Silhouette Score
    silhouette = silhouette_score(df[feature_columns], df['cluster'])

    # Compute confusion matrix and metrics
    true_labels = (df[target_column] == "Sb2").astype(int)
    pred_labels = df['cluster']
    try:
        tn, fp, fn, tp = confusion_matrix(true_labels, pred_labels, labels=[0, 1]).ravel()
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    except ValueError:
        recall = specificity = precision = f1_score = 0

    return {
        "Cohesion": cohesion,
        "Separation": separation,
        "Silhouette Score": silhouette,
        "Recall": recall,
        "Specificity": specificity,
        "F1-Score": f1_score
    }

def plot_metrics(results, metric_name):
    """Plot and save a bar chart for the specified metric."""
    plt.figure(figsize=(10, 6))
    datasets = list(results.keys())
    values = [metrics[metric_name] for metrics in results.values()]
    
    sns.barplot(x=datasets, y=values, palette="viridis")
    plt.title(f"Comparison of {metric_name}")
    plt.xlabel("Datasets")
    plt.ylabel(metric_name)
    plt.xticks(rotation=45, ha="right")
    
    for i, value in enumerate(values):
        plt.text(i, value, f"{value:.2f}", ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    if not os.path.exists("output/cluster"):
        os.makedirs("output/cluster")
    plot_file = f"output/cluster/{module_name}_metric_{metric_name}.png"
    plt.savefig(plot_file)
    print(f"Saved {metric_name} plot as {plot_file}")
    plt.close()

def plot_3d_combinations(df, feature_columns, file_name, kmeans):
    """Visualize clusters in 3D for all feature combinations with custom cluster colors and legends."""
    if len(feature_columns) < 3:
        print("Not enough features to plot 3D combinations.")
        return

    feature_combinations = combinations(feature_columns, 3)
    
    for comb in feature_combinations:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        cluster_0 = df[df['cluster'] == 0]
        cluster_1 = df[df['cluster'] == 1]
        
        ax.scatter(cluster_0[comb[0]], cluster_0[comb[1]], cluster_0[comb[2]], c='red', label='C1', alpha=0.7)
        ax.scatter(cluster_1[comb[0]], cluster_1[comb[1]], cluster_1[comb[2]], c='green', label='C2', alpha=0.7)

        ax.set_xlabel(comb[0])
        ax.set_ylabel(comb[1])
        ax.set_zlabel(comb[2])

        plt.title(f"3D Cluster Visualization for {file_name} ({comb[0]} vs {comb[1]} vs {comb[2]})")
        ax.legend(loc='upper right')

        plt.tight_layout()

        if not os.path.exists("output/cluster"):
            os.makedirs("output/cluster")
        plot_file = f"output/cluster/{module_name}_3d_combination_{file_name}_{comb[0]}_{comb[1]}_{comb[2]}.png"
        plt.savefig(plot_file)
        print(f"Saved 3D combination plot for {file_name} ({comb[0]} vs {comb[1]} vs {comb[2]}) as {plot_file}")
        plt.close()

def hierarchical_clustering(df, feature_columns):
    """Apply hierarchical clustering with a threshold for splitting into exactly 2 clusters."""
    agglomerative = AgglomerativeClustering(n_clusters=2, linkage='ward')
    Z = linkage(df[feature_columns], method='ward')
    plt.figure(figsize=(10, 6))
    dendrogram(Z)
    plt.title("Hierarchical Clustering Dendrogram")
    plt.xlabel("Samples")
    plt.ylabel("Distance")

    if not os.path.exists("output/cluster"):
        os.makedirs("output/cluster")
    dendrogram_file = f"output/cluster/{module_name}_dendrogram.png"
    plt.savefig(dendrogram_file)
    print(f"Saved dendrogram plot as {dendrogram_file}")
    plt.close()

    df['hierarchical_cluster'] = agglomerative.fit_predict(df[feature_columns])
    return df, agglomerative

def save_clustered_data(df, file_name):
    """Save the clustered data to a CSV file."""
    if not os.path.exists("output/cluster"):
        os.makedirs("output/cluster")
    output_file = f"output/cluster/{module_name}_clustered_{file_name}.csv"
    df.to_csv(output_file, index=False)
    print(f"Saved clustered data as {output_file}")



# Clustering ends --------------------------------------------------

#%% MAIN CODE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if __name__ == "__main__":
    # Feature generation
    for channel in ['M1', 'M2', 'CPz']:
        train_validate, test = create_datasets('sb1', 'se1', 'sb2', 'se2', channel)
        save_and_plot(train_validate, test, channel)

    print("Feature generation completed.")
    print(f"\"{module_name}\" module begins.")

    # Load data
    data_frames = load_data(DATA_FILES)
    results = {}
    kmeans_models = {}

    train_validate_data = [file_name for file_name in data_frames if 'TrainValidate' in file_name]
    test_data = [file_name for file_name in data_frames if 'TestData' in file_name]

    for file_name, df in data_frames.items():
        print(f"Processing {file_name}...")

        # Extract feature columns
        feature_columns = [col for col in df.columns if col.startswith('a')]
        target_column = "target"

        # Preprocess the data
        print(f"Preprocessing data for {file_name}...")
        # Use standarize_data with PCA enabled
        df_preprocessed = preprocess_data(df)
        # df_preprocessed = standarize_data(df, feature_columns, apply_pca=True)


        # Clustering
        print(f"Clustering for {file_name}...")
        if len(df_preprocessed.columns) < 3:
            print(f"Skipping {file_name} due to insufficient data after preprocessing.")
            continue

# using all features -----------------
        if file_name in train_validate_data:
            kmeans = KMeans(n_clusters=2, random_state=42)
            df_preprocessed['cluster'] = kmeans.fit_predict(df_preprocessed[feature_columns])

            kmeans_models[file_name] = kmeans
            metrics = calculate_metrics(df_preprocessed, feature_columns, target_column)
            results[file_name] = metrics

            plot_3d_combinations(df_preprocessed, feature_columns, file_name, kmeans)
            df_hierarchical, agglomerative = hierarchical_clustering(df_preprocessed, feature_columns)
            save_clustered_data(df_hierarchical, file_name)

        if file_name in test_data:
            print(f"Testing {file_name}...")
            if file_name in kmeans_models:
                kmeans = kmeans_models[file_name]
                df_test = preprocess_data(df)
                df_test['cluster'] = kmeans.predict(df_test[feature_columns])
                save_clustered_data(df_test, f"test_{file_name}")


# Using PCA------------------------
        # if file_name in train_validate_data:
        #     kmeans = KMeans(n_clusters=2, random_state=42)
        #     df_preprocessed['cluster'] = kmeans.fit_predict(df_preprocessed[['PCA1', 'PCA2', 'PCA3']])

        #     kmeans_models[file_name] = kmeans
        #     metrics = calculate_metrics(df_preprocessed, ['PCA1', 'PCA2', 'PCA3'], "target")
        #     results[file_name] = metrics

        #     plot_3d_combinations(df_preprocessed, ['PCA1', 'PCA2', 'PCA3'], file_name, kmeans)
        #     df_hierarchical, agglomerative = hierarchical_clustering(df_preprocessed, ['PCA1', 'PCA2', 'PCA3'])
        #     save_clustered_data(df_hierarchical, file_name)

        # if file_name in test_data:
        #     print(f"Testing {file_name}...")
        #     if file_name in kmeans_models:
        #         kmeans = kmeans_models[file_name]
        #         df_test_preprocessed = standarize_data(df, feature_columns, apply_pca=True)
        #         df_test_preprocessed['cluster'] = kmeans.predict(df_test_preprocessed[['PCA1', 'PCA2', 'PCA3']])
        #         save_clustered_data(df_test_preprocessed, f"test_{file_name}")
# Using PCA ends ------------------------



    # Plot metrics
    metrics_to_plot = ["Cohesion", "Separation", "Silhouette Score", "Recall", "Specificity", "F1-Score"]
    for metric in metrics_to_plot:
        plot_metrics(results, metric)

    print(f"\"{module_name}\" module ends.")


#ends ------------------------------------------------------------------
