import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import cross_val_score, KFold

def summarise_data(filename):
    # Load the dataset
    try:
        df = pd.read_csv(filename)
    except Exception as e:
        print(f"Error reading the file: {e}")
        return

    # Basic dataset overview
    print(f"\n--- Dataset Overview ---")
    print(f"1. Number of rows and columns: {df.shape}")
    print(f"2. Columns and their data types:\n{df.dtypes}")
    
    # Summary statistics for numerical columns
    print("\n3. Summary statistics for numerical columns:")
    print(df.describe())
    
    # Check for missing values
    print("\n4. Missing values in each column:")
    print(df.isnull().sum())
    
    # Check for unique values in each column
    print("\n5. Unique values in each column:")
    for column in df.columns:
        print(f"{column}: {df[column].nunique()} unique values")
    
    # Additional Information
    print("\n6. Sample of the dataset:")
    print(df.head())  # Print first 5 rows as a sample


def correlation_heatmap(filename):
    # Load the dataset
    try:
        df = pd.read_csv(filename)
    except Exception as e:
        print(f"Error reading the file: {e}")
        return

    # Compute the Spearman correlation matrix
    corr_matrix = df.corr(method='spearman')

    # Create a heatmap for the correlation matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, cbar=True)
    plt.title("Spearman Rank Correlation Heatmap")
    plt.show()


def handle_missing_data(df):
    # Fill missing values with the median (simple method)
    df_filled = df.fillna(df.median())
    print("\n--- Missing Data Handling ---")
    print("Missing data handled by filling with median.")
    return df_filled


def detect_outliers(df):
    # Using Z-Score method to detect outliers
    print("\n--- Outlier Detection ---")
    z_scores = stats.zscore(df.select_dtypes(include=[float, int]))
    abs_z_scores = np.abs(z_scores)
    outliers = (abs_z_scores > 3).all(axis=1)
    print(f"Outliers detected: {np.sum(outliers)} out of {df.shape[0]} rows.")
    return df[~outliers]  # Removing outliers


def normalize_data(df):
    # Min-Max Normalization
    print("\n--- Data Normalization ---")
    scaler = MinMaxScaler()
    normalized_df = pd.DataFrame(scaler.fit_transform(df.select_dtypes(include=[float, int])), columns=df.select_dtypes(include=[float, int]).columns)
    return normalized_df


def standardize_data(df):
    # Z-Score Standardization
    print("\n--- Data Standardization ---")
    scaler = StandardScaler()
    standardized_df = pd.DataFrame(scaler.fit_transform(df.select_dtypes(include=[float, int])), columns=df.select_dtypes(include=[float, int]).columns)
    return standardized_df


def perform_kmeans_clustering(df, n_clusters=3):
    # K-Means Clustering
    print(f"\n--- K-Means Clustering ---")
    kmeans = KMeans(n_clusters=n_clusters)
    df['cluster'] = kmeans.fit_predict(df.select_dtypes(include=[float, int]))
    print(f"K-Means Clustering completed with {n_clusters} clusters.")
    return df


def hierarchical_clustering(df):
    # Hierarchical Clustering Dendrogram
    from scipy.cluster.hierarchy import dendrogram, linkage
    print("\n--- Hierarchical Clustering ---")
    linked = linkage(df.select_dtypes(include=[float, int]), method='ward')
    plt.figure(figsize=(10, 7))
    dendrogram(linked)
    plt.title("Hierarchical Clustering Dendrogram")
    plt.show()


def hypothesis_testing(df, col1, col2):
    # T-Test between two columns
    print(f"\n--- Hypothesis Testing (T-Test) ---")
    t_stat, p_val = stats.ttest_ind(df[col1].dropna(), df[col2].dropna())
    print(f"T-statistic: {t_stat}, P-value: {p_val}")
    if p_val < 0.05:
        print("Result is statistically significant.")
    else:
        print("Result is not statistically significant.")


def evaluate_model(df, model, X, y):
    # Perform K-Fold Cross Validation
    print("\n--- Model Evaluation ---")
    kf = KFold(n_splits=5)
    scores = cross_val_score(model, X, y, cv=kf)
    print(f"Model Cross Validation Scores: {scores}")
    print(f"Average Cross Validation Score: {scores.mean()}")


def main():
    # Ask the user for the filename
    filename = input("Enter the filename of the dataset (CSV format): ").strip()

    # Call the summarise_data function
    summarise_data(filename)

    # Handle missing data
    df = pd.read_csv(filename)
    df = handle_missing_data(df)

    # Detect and remove outliers
    df = detect_outliers(df)

    # Normalize data
    df_normalized = normalize_data(df)

    # Standardize data
    df_standardized = standardize_data(df)

    # Perform clustering
    df = perform_kmeans_clustering(df)

    # Display hierarchical clustering dendrogram
    hierarchical_clustering(df)

    # Hypothesis testing example (change column names as needed)
    col1 = input("\nEnter the name of the first column for t-test: ").strip()
    col2 = input("Enter the name of the second column for t-test: ").strip()
    hypothesis_testing(df, col1, col2)

    # Ask user if they want the correlation heatmap
    generate_heatmap = input("\nWould you like to see the correlation heatmap? (yes/no): ").strip().lower()
    if generate_heatmap == 'yes':
        correlation_heatmap(filename)


if __name__ == "__main__":
    main()
