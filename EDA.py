import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Let's start by loading the data
df = pd.read_csv("Healthcare Ransomware Dataset.csv")

# Getting to know the dataset a bit
print("📄 A quick look at the dataset structure:")
print(df.info())

print("\n🧾 Here's a peek at the first few rows:")
print(df.head())

print("\n📊 Some basic summary stats (for all columns):")
print(df.describe(include='all'))

# Making sure the 'attack_date' column is recognized as a date
df['attack_date'] = pd.to_datetime(df['attack_date'])

# Let's check if there are any missing values we should be aware of
print("\n🔍 Checking for missing values:")
print(df.isnull().sum())

# --- Visual Explorations ---

# 1. How common are different infection rates?
plt.figure(figsize=(8, 4))
sns.histplot(df['ransomware_infection_rate_(%)'], kde=True, bins=30)
plt.title('How Often Different Ransomware Infection Rates Occur')
plt.xlabel('Infection Rate (%)')
plt.ylabel('Number of Cases')
plt.tight_layout()
plt.show()

# 2. How long it takes to recover, grouped by organization type
plt.figure(figsize=(10, 5))
sns.boxplot(x='org_type', y='recovery_time_(days)', data=df)
plt.title('Recovery Time by Organization Type')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 3. Which entry methods are most common?
plt.figure(figsize=(8, 4))
sns.countplot(data=df, y='entry_method', order=df['entry_method'].value_counts().index)
plt.title('How Attackers Get In (Entry Method Counts)')
plt.tight_layout()
plt.show()

# 4. Is paying the ransom linked to more data being restored?
plt.figure(figsize=(6, 4))
sns.boxplot(x='paid_ransom', y='data_restored', data=df)
plt.title('Data Restored (%) Based on Whether Ransom Was Paid')
plt.xlabel('Paid Ransom')
plt.ylabel('Data Restored (%)')
plt.tight_layout()
plt.show()

# 5. How are the numeric features related to each other?
plt.figure(figsize=(10, 6))
numeric_df = df.select_dtypes(include=['int64', 'float64'])
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Looking at Correlations Between Numeric Columns')
plt.tight_layout()
plt.show()
