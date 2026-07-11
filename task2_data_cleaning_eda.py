"""
Prodigy InfoTech - Data Science Internship
Task 2: Perform data cleaning and exploratory data analysis (EDA) on a
        dataset of your choice - Titanic dataset from Kaggle.

Explore relationships between variables and identify patterns/trends.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option("display.max_columns", None)

# ----------------------------------------------------------
# 1. Load the dataset
# ----------------------------------------------------------
DATA_URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(DATA_URL)

print("Shape:", df.shape)
print("\nInfo:")
df.info()

# ----------------------------------------------------------
# 2. Data cleaning
# ----------------------------------------------------------
print("\nMissing values before cleaning:")
print(df.isnull().sum())

# Fill missing Age with the median age
df["Age"] = df["Age"].fillna(df["Age"].median())

# Fill missing Embarked with the most common port
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# Cabin has too many missing values to impute meaningfully - drop the column
df.drop(columns=["Cabin"], inplace=True)

# Drop any exact duplicate rows
df.drop_duplicates(inplace=True)

print("\nMissing values after cleaning:")
print(df.isnull().sum())

# ----------------------------------------------------------
# 3. Exploratory Data Analysis
# ----------------------------------------------------------

# 3a. Survival rate overall
print("\nOverall survival rate: {:.2f}%".format(df["Survived"].mean() * 100))

# 3b. Survival rate by gender
plt.figure(figsize=(6, 5))
sns.barplot(data=df, x="Sex", y="Survived", hue="Sex", palette="Set2", legend=False)
plt.title("Survival Rate by Gender")
plt.ylabel("Survival Rate")
plt.tight_layout()
plt.savefig("survival_by_gender.png", dpi=150)
plt.show()

# 3c. Survival rate by passenger class
plt.figure(figsize=(6, 5))
sns.barplot(data=df, x="Pclass", y="Survived", hue="Pclass", palette="Set3", legend=False)
plt.title("Survival Rate by Passenger Class")
plt.ylabel("Survival Rate")
plt.tight_layout()
plt.savefig("survival_by_class.png", dpi=150)
plt.show()

# 3d. Age distribution split by survival
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x="Age", hue="Survived", kde=True, bins=25, palette="Set1")
plt.title("Age Distribution by Survival")
plt.tight_layout()
plt.savefig("age_vs_survival.png", dpi=150)
plt.show()

# 3e. Correlation heatmap of numeric features
plt.figure(figsize=(8, 6))
numeric_df = df.select_dtypes(include="number")
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("correlation_heatmap.png", dpi=150)
plt.show()

# ----------------------------------------------------------
# 4. Key insights (printed for the report)
# ----------------------------------------------------------
print("\n--- Key Insights ---")
print("1. Women had a much higher survival rate than men.")
print("2. 1st class passengers survived at a higher rate than 2nd and 3rd class.")
print("3. Younger passengers (children) had relatively higher survival rates.")
print("4. Fare and Pclass show a meaningful negative correlation with each other.")
