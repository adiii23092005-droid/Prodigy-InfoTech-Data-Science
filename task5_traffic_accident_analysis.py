"""
Prodigy InfoTech - Data Science Internship
Task 5: Analyze traffic accident data to identify patterns related to
        road conditions, weather, and time of day. Visualize accident
        hotspots and contributing factors.

Dataset: US Accidents (Sobhan Moosavi) - Kaggle
https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents

NOTE ON THE DATASET
--------------------
This dataset is very large (2M+ rows, several GB) and requires a Kaggle
account to download - it can't be streamed directly with pd.read_csv like
the earlier tasks. Steps to get the data:

1. Install the Kaggle CLI:      pip install kaggle
2. Get your API token from https://www.kaggle.com/settings -> "Create New Token"
   (this downloads kaggle.json - place it in ~/.kaggle/kaggle.json)
3. Download the dataset:
       kaggle datasets download -d sobhanmoosavi/us-accidents
       unzip us-accidents.zip
4. Place the resulting CSV (e.g. "US_Accidents_March23.csv") in the same
   folder as this script, and set CSV_PATH below to match its filename.

Because the file is huge, this script reads it in a memory-friendly way
(only the columns we need) and works on a random sample for the plots
that would otherwise be too slow/heavy to render.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------------------------------------
# 1. Load the dataset (only relevant columns, to save memory)
# ----------------------------------------------------------
CSV_PATH = "US_Accidents_March23.csv"  # <-- update to your downloaded filename

USE_COLS = [
    "Start_Time", "Severity", "Start_Lat", "Start_Lng",
    "Weather_Condition", "Temperature(F)", "Visibility(mi)",
    "Wind_Speed(mph)", "State", "City",
]

df = pd.read_csv(CSV_PATH, usecols=USE_COLS)
print("Full shape:", df.shape)

# Work on a manageable random sample for visualization
SAMPLE_SIZE = 200_000
df_sample = df.sample(n=min(SAMPLE_SIZE, len(df)), random_state=42)

# ----------------------------------------------------------
# 2. Cleaning
# ----------------------------------------------------------
df_sample = df_sample.dropna(subset=["Start_Time", "Weather_Condition", "Start_Lat", "Start_Lng"])
df_sample["Start_Time"] = pd.to_datetime(df_sample["Start_Time"], errors="coerce")
df_sample = df_sample.dropna(subset=["Start_Time"])

df_sample["Hour"] = df_sample["Start_Time"].dt.hour
df_sample["DayOfWeek"] = df_sample["Start_Time"].dt.day_name()

# ----------------------------------------------------------
# 3. Accidents by hour of day
# ----------------------------------------------------------
plt.figure(figsize=(10, 5))
sns.countplot(data=df_sample, x="Hour", hue="Hour", palette="mako", legend=False)
plt.title("Number of Accidents by Hour of Day")
plt.xlabel("Hour (24h)")
plt.ylabel("Accident Count")
plt.tight_layout()
plt.savefig("accidents_by_hour.png", dpi=150)
plt.show()

# ----------------------------------------------------------
# 4. Accidents by day of week
# ----------------------------------------------------------
day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
plt.figure(figsize=(9, 5))
sns.countplot(data=df_sample, x="DayOfWeek", order=day_order, hue="DayOfWeek",
              palette="crest", legend=False)
plt.title("Number of Accidents by Day of Week")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("accidents_by_day.png", dpi=150)
plt.show()

# ----------------------------------------------------------
# 5. Top weather conditions during accidents
# ----------------------------------------------------------
top_weather = df_sample["Weather_Condition"].value_counts().head(10)

plt.figure(figsize=(9, 6))
sns.barplot(x=top_weather.values, y=top_weather.index, hue=top_weather.index,
            palette="flare", legend=False)
plt.title("Top 10 Weather Conditions During Accidents")
plt.xlabel("Accident Count")
plt.tight_layout()
plt.savefig("top_weather_conditions.png", dpi=150)
plt.show()

# ----------------------------------------------------------
# 6. Severity distribution
# ----------------------------------------------------------
plt.figure(figsize=(6, 5))
sns.countplot(data=df_sample, x="Severity", hue="Severity", palette="rocket", legend=False)
plt.title("Accident Severity Distribution")
plt.xlabel("Severity (1 = least, 4 = most severe)")
plt.tight_layout()
plt.savefig("severity_distribution.png", dpi=150)
plt.show()

# ----------------------------------------------------------
# 7. Accident hotspots (scatter map using lat/lng)
# ----------------------------------------------------------
plt.figure(figsize=(10, 7))
hotspot_sample = df_sample.sample(n=min(20_000, len(df_sample)), random_state=42)
plt.scatter(
    hotspot_sample["Start_Lng"], hotspot_sample["Start_Lat"],
    s=2, alpha=0.3, c=hotspot_sample["Severity"], cmap="Reds"
)
plt.colorbar(label="Severity")
plt.title("Accident Hotspots (by Location and Severity)")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.tight_layout()
plt.savefig("accident_hotspots.png", dpi=150)
plt.show()

# ----------------------------------------------------------
# 8. Summary
# ----------------------------------------------------------
print("\n--- Key Insights ---")
print("Peak accident hour:", df_sample["Hour"].value_counts().idxmax())
print("Most common day for accidents:", df_sample["DayOfWeek"].value_counts().idxmax())
print("Most common weather during accidents:", top_weather.index[0])
print("Most common severity level:", df_sample["Severity"].value_counts().idxmax())
