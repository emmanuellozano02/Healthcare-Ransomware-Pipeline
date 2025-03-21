import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(filepath):
    """Load the dataset from a CSV file."""
    try:
        df = pd.read_csv(filepath)
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def basic_info(df):
    """Print basic information about the dataset."""
    print("📄 A quick look at the dataset structure:")
    print(df.info())
    print("\n🧾 Here's a peek at the first few rows:")
    print(df.head())

def summary_stats(df):
    """Print summary statistics of the dataset."""
    print("\n📊 Some basic summary stats (for all columns):")
    print(df.describe(include='all'))

def check_missing_values(df):
    """Check for missing values in the dataset."""
    print("\n🔍 Checking for missing values:")
    print(df.isnull().sum())

def plot_infection_rate(df):
    """Plot the distribution of ransomware infection rates."""
    plt.figure(figsize=(8, 4))
    sns.histplot(df['ransomware_infection_rate_(%)'], kde=True, bins=30)
    plt.title('How Often Different Ransomware Infection Rates Occur')
    plt.xlabel('Infection Rate (%)')
    plt.ylabel('Number of Cases')
    plt.tight_layout()
    plt.show()

def plot_recovery_time(df):
    """Plot recovery time by organization type."""
    plt.figure(figsize=(10, 5))
    sns.boxplot(x='org_type', y='recovery_time_(days)', data=df)
    plt.title('Recovery Time by Organization Type')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_entry_method(df):
    """Plot entry method counts."""
    plt.figure(figsize=(8, 4))
    sns.countplot(data=df, y='entry_method', order=df['entry_method'].value_counts().index)
    plt.title('How Attackers Get In (Entry Method Counts)')
    plt.tight_layout()
    plt.show()

def plot_data_restoration(df):
    """Plot data restoration based on whether ransom was paid."""
    plt.figure(figsize=(6, 4))
    sns.boxplot(x='paid_ransom', y='data_restored', data=df)
    plt.title('Data Restored (%) Based on Whether Ransom Was Paid')
    plt.xlabel('Paid Ransom')
    plt.ylabel('Data Restored (%)')
    plt.tight_layout()
    plt.show()

def plot_correlation_matrix(df):
    """Plot the correlation matrix of numeric columns."""
    plt.figure(figsize=(10, 6))
    numeric_df = df.select_dtypes(include=['int64', 'float64'])
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Looking at Correlations Between Numeric Columns')
    plt.tight_layout()
    plt.show()

def main():
    """Main function to run the script."""
    filepath = "Healthcare Ransomware Dataset.csv"
    df = load_data(filepath)

    if df is not None:
        # Convert dates
        df['attack_date'] = pd.to_datetime(df['attack_date'])

        # Data exploration and visualization
        basic_info(df)
        summary_stats(df)
        check_missing_values(df)
        plot_infection_rate(df)
        plot_recovery_time(df)
        plot_entry_method(df)
        plot_data_restoration(df)
        plot_correlation_matrix(df)

if __name__ == "__main__":
    main()
