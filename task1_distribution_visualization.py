"""
Prodigy InfoTech - Data Science Internship
Task 1: Visualize the distribution of a categorical or continuous variable
        (Age - continuous, Gender/Sex - categorical)

Dataset: Titanic dataset (has both Age and Sex columns, publicly available)
You can swap the DATA_URL below for the Task-1 dataset link Prodigy gave you
if you'd rather use that one - the code works the same way.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------------------------------------
# 1. Load the dataset
# ----------------------------------------------------------
DATA_URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(DATA_URL)

print("Shape of dataset:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

print("\nColumn info:")
print(df.info())

# ----------------------------------------------------------
# 2. Basic cleaning (only what's needed for this task)
# ----------------------------------------------------------
# Age has some missing values - drop them just for the age plot
age_data = df["Age"].dropna()

# ----------------------------------------------------------
# 3. Histogram - distribution of Age (continuous variable)
# ----------------------------------------------------------
plt.figure(figsize=(8, 5))
sns.histplot(age_data, bins=20, kde=True, color="steelblue", edgecolor="black")
plt.title("Distribution of Passenger Age (Titanic Dataset)", fontsize=14)
plt.xlabel("Age")
plt.ylabel("Number of Passengers")
plt.tight_layout()
plt.savefig("age_distribution.png", dpi=150)
plt.show()

# ----------------------------------------------------------
# 4. Bar chart - distribution of Gender (categorical variable)
# ----------------------------------------------------------
plt.figure(figsize=(6, 5))
sns.countplot(data=df, x="Sex", hue="Sex", palette="Set2", legend=False)
plt.title("Distribution of Passenger Gender (Titanic Dataset)", fontsize=14)
plt.xlabel("Gender")
plt.ylabel("Number of Passengers")
plt.tight_layout()
plt.savefig("gender_distribution.png", dpi=150)
plt.show()

# ----------------------------------------------------------
# 5. Quick summary printed to console
# ----------------------------------------------------------
print("\n--- Age summary ---")
print(age_data.describe())

print("\n--- Gender counts ---")
print(df["Sex"].value_counts())
