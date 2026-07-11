"""
Prodigy InfoTech - Data Science Internship
Task 3: Build a decision tree classifier to predict whether a customer
        will purchase a product/service, based on demographic and
        behavioral data - Bank Marketing dataset (UCI).
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# ----------------------------------------------------------
# 1. Load the dataset
# ----------------------------------------------------------
DATA_URL = "https://raw.githubusercontent.com/justmarkham/DAT8/master/data/bank-additional.csv"
df = pd.read_csv(DATA_URL, sep=";")

print("Shape:", df.shape)
print(df.head())

# ----------------------------------------------------------
# 2. Basic cleaning
# ----------------------------------------------------------
# The target column is 'y' (yes/no -> did the client subscribe?)
df.drop_duplicates(inplace=True)

# ----------------------------------------------------------
# 3. Encode categorical variables
# ----------------------------------------------------------
df_encoded = df.copy()
label_encoders = {}

categorical_cols = df_encoded.select_dtypes(include="object").columns

for col in categorical_cols:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_encoded[col])
    label_encoders[col] = le

# ----------------------------------------------------------
# 4. Train / test split
# ----------------------------------------------------------
X = df_encoded.drop(columns=["y"])
y = df_encoded["y"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ----------------------------------------------------------
# 5. Train the Decision Tree
# ----------------------------------------------------------
clf = DecisionTreeClassifier(max_depth=5, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

# ----------------------------------------------------------
# 6. Evaluate
# ----------------------------------------------------------
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No", "Yes"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.show()

# ----------------------------------------------------------
# 7. Feature importance
# ----------------------------------------------------------
importances = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False)

plt.figure(figsize=(8, 6))
sns.barplot(x=importances.values[:10], y=importances.index[:10], hue=importances.index[:10],
            palette="viridis", legend=False)
plt.title("Top 10 Most Important Features")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150)
plt.show()

# ----------------------------------------------------------
# 8. Visualize the decision tree (top levels)
# ----------------------------------------------------------
plt.figure(figsize=(20, 10))
plot_tree(
    clf,
    feature_names=X.columns,
    class_names=["No", "Yes"],
    filled=True,
    max_depth=3,
    fontsize=8,
)
plt.title("Decision Tree (first 3 levels)")
plt.tight_layout()
plt.savefig("decision_tree.png", dpi=150)
plt.show()

print("\nTop 5 important features:")
print(importances.head(5))
