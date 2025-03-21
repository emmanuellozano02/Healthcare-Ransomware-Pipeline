import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load
df = pd.read_csv("Healthcare Ransomware Dataset.csv")

# Info
print("📄 A quick look at the dataset structure:")
print(df.info())

# Preview
print("\n🧾 Here's a peek at the first few rows:")
print(df.head())

# Stats
print("\n📊 Some basic summary stats (for all columns):")
print(df.describe(include='all'))

# Dates
df['attack_date'] = pd.to_datetime(df['attack_date'])

# Missing
print("\n🔍 Checking for missing values:")
print(df.isnull().sum())

# --- Plots ---

# Infection
plt.figure(figsize=(8, 4))
sns.histplot(df['ransomware_infection_rate_(%)'], kde=True, bins=30)
plt.title('How Often Different Ransomware Infection Rates Occur')
plt.xlabel('Infection Rate (%)')
plt.ylabel('Number of Cases')
plt.tight_layout()
plt.show()

# Recovery
plt.figure(figsize=(10, 5))
sns.boxplot(x='org_type', y='recovery_time_(days)', data=df)
plt.title('Recovery Time by Organization Type')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Entry
plt.figure(figsize=(8, 4))
sns.countplot(data=df, y='entry_method', order=df['entry_method'].value_counts().index)
plt.title('How Attackers Get In (Entry Method Counts)')
plt.tight_layout()
plt.show()

# Ransom
plt.figure(figsize=(6, 4))
sns.boxplot(x='paid_ransom', y='data_restored', data=df)
plt.title('Data Restored (%) Based on Whether Ransom Was Paid')
plt.xlabel('Paid Ransom')
plt.ylabel('Data Restored (%)')
plt.tight_layout()
plt.show()

# Correlation
plt.figure(figsize=(10, 6))
numeric_df = df.select_dtypes(include=['int64', 'float64'])
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Looking at Correlations Between Numeric Columns')
plt.tight_layout()
plt.show()
