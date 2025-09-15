import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Task 1: Load and Explore Dataset
def load_dataset():
    try:
        # Load iris dataset from CSV file
        df = pd.read_csv("iris_dataset.csv")
        return df
    except FileNotFoundError:
        print("Error: Dataset file not found. Make sure 'iris_dataset.csv' is in the same folder.")
        return None

def explore_data(df):
    print("\nFirst 5 rows of the dataset:")
    print(df.head())

    print("\nDataset info:")
    print(df.info())

    print("\nMissing values:")
    print(df.isnull().sum())

    # Cleaning: Fill or drop missing values (if any)
    df = df.dropna()
    return df
# ----------------------------

# Task 2: Basic Data Analysis
def analyze_data(df):
    print("\nDescriptive Statistics:")
    print(df.describe())

    # Group by species and compute mean
    grouped = df.groupby("target").mean(numeric_only=True)
    print("\nMean values grouped by species:")
    print(grouped)
    
    return grouped
# ----------------------------

# Task 3: Data Visualization
def visualize_data(df):
    # Line chart (cumulative sum of sepal length as trend example)
    plt.figure(figsize=(8,5))
    df_sorted = df.sort_index()
    df_sorted["sepal length (cm)"].cumsum().plot()
    plt.title("Cumulative Sepal Length Over Samples")
    plt.xlabel("Sample Index")
    plt.ylabel("Cumulative Sepal Length (cm)")
    plt.grid(True)
    plt.show()

    # Bar chart (average petal length per species)
    plt.figure(figsize=(8,5))
    sns.barplot(x="target", y="petal length (cm)", data=df, ci=None)
    plt.title("Average Petal Length per Species")
    plt.xlabel("Species")
    plt.ylabel("Petal Length (cm)")
    plt.show()

    # Histogram (distribution of sepal width)
    plt.figure(figsize=(8,5))
    plt.hist(df["sepal width (cm)"], bins=15, color="skyblue", edgecolor="black")
    plt.title("Distribution of Sepal Width")
    plt.xlabel("Sepal Width (cm)")
    plt.ylabel("Frequency")
    plt.show()

    # Scatter plot (sepal length vs. petal length)
    plt.figure(figsize=(8,5))
    sns.scatterplot(x="sepal length (cm)", y="petal length (cm)", hue="target", data=df, palette="Set1")
    plt.title("Sepal Length vs Petal Length")
    plt.xlabel("Sepal Length (cm)")
    plt.ylabel("Petal Length (cm)")
    plt.legend(title="Species")
    plt.show()
# ----------------------------

# Findings / Observations
def findings():
    print("\nKey Findings:")
    print("1. The dataset contains 150 samples of iris flowers with 4 features each.")
    print("2. There are no missing values in this dataset.")
    print("3. Statistical summaries show distinct differences in feature ranges among species.")
    print("4. Average petal length clearly varies across species, useful for classification.")
    print("5. Scatter plot indicates a strong positive correlation between sepal length and petal length.")
# ----------------------------

# Main Execution
if __name__ == "__main__":
    df = load_dataset()
    if df is not None:
        df = explore_data(df)
        analyze_data(df)
        visualize_data(df)
        findings()