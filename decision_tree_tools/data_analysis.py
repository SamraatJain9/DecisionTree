import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def summarise_data(filename):
    # Load the dataset
    try:
        df = pd.read_csv(filename)
    except Exception as e:
        print(f"Error reading the file: {e}")
        return

    # Basic dataset overview
    print(f"Dataset Overview:\n")
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


def main():
    # Ask the user for the filename
    filename = input("Enter the filename of the dataset (CSV format): ").strip()

    # Call the summarise_data function
    summarise_data(filename)

    # Ask user if they want the correlation heatmap
    generate_heatmap = input("\nWould you like to see the correlation heatmap? (yes/no): ").strip().lower()
    if generate_heatmap == 'yes':
        correlation_heatmap(filename)


if __name__ == "__main__":
    main()
