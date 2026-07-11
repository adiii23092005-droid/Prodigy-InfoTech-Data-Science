# Prodigy InfoTech - Data Science Internship

This repository contains all tasks completed as part of the **Data Science Internship** at **Prodigy InfoTech**.

## 📌 Tasks

| Task | Description | Status |
|------|-------------|--------|
| [Task 1](./Task-1) | Visualize the distribution of a categorical or continuous variable (bar chart / histogram) | ✅ Completed |
| [Task 2](./Task-2) | Data cleaning and exploratory data analysis (EDA) on the Titanic dataset | ✅ Completed |
| [Task 3](./Task-3) | Decision tree classifier to predict customer purchase behavior (Bank Marketing dataset) | ✅ Completed |
| [Task 4](./Task-4) | Sentiment analysis on social media data (tweets) | ✅ Completed |
| [Task 5](./Task-5) | Traffic accident data analysis - hotspots, weather, time-of-day patterns | ✅ Completed |

## 🛠️ Tech Stack

- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn (Decision Tree Classifier)
- NLTK (VADER sentiment analysis)

## 📂 Repository Structure

```
Prodigy-InfoTech-Data-Science/
│
├── Task-1/
│   └── task1_distribution_visualization.py
│
├── Task-2/
│   └── task2_data_cleaning_eda.py
│
├── Task-3/
│   └── task3_decision_tree_classifier.py
│
├── Task-4/
│   └── task4_sentiment_analysis.py
│
├── Task-5/
│   └── task5_traffic_accident_analysis.py
│
└── README.md
```

## 📖 Task Details

### Task 1 - Distribution Visualization
Visualized the distribution of Age (continuous) and Gender (categorical) from the Titanic dataset using a histogram and a bar chart.

### Task 2 - Data Cleaning & EDA
Cleaned the Titanic dataset (handled missing values, dropped duplicates) and explored relationships between survival, gender, passenger class, and age using bar charts, histograms, and a correlation heatmap.

### Task 3 - Decision Tree Classifier
Built a decision tree model on the UCI Bank Marketing dataset to predict whether a customer subscribes to a term deposit, based on demographic and behavioral features. Evaluated with accuracy, a classification report, a confusion matrix, and feature importance.

### Task 4 - Sentiment Analysis
Analyzed sentiment in social media posts (tweets) using NLTK's VADER lexicon-based analyzer. Visualized sentiment distribution, compound score spread, and the most common words in positive vs. negative posts.

### Task 5 - Traffic Accident Analysis
Analyzed the US Accidents dataset (Kaggle) to identify patterns by hour of day, day of week, weather condition, and severity, and plotted accident hotspots on a scatter map by location.

## ▶️ How to Run

Each task is a standalone Python script:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn nltk
python Task-1/task1_distribution_visualization.py
```

Tasks 1-4 load their datasets directly from public URLs, so no manual download is needed. Task 5 requires manually downloading the US Accidents dataset from Kaggle (see the comment at the top of the script for instructions).

## 🙋 About

Internship: Data Science Track (DS)
Organization: [Prodigy InfoTech](https://www.prodigyinfotech.dev/)
