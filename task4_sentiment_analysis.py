"""
Prodigy InfoTech - Data Science Internship
Task 4: Analyze and visualize sentiment patterns in social media data to
        understand public opinion and attitudes towards specific topics.

Uses NLTK's built-in twitter_samples corpus (real tweets) and VADER
(a lexicon-based sentiment analyzer well-suited to short, informal social
media text). No external dataset download required beyond NLTK's own
data packages.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk

# ----------------------------------------------------------
# 1. Download required NLTK resources (one-time)
# ----------------------------------------------------------
nltk.download("twitter_samples")
nltk.download("vader_lexicon")
nltk.download("stopwords")

from nltk.corpus import twitter_samples, stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# ----------------------------------------------------------
# 2. Load sample social media data
# ----------------------------------------------------------
positive_tweets = twitter_samples.strings("positive_tweets.json")
negative_tweets = twitter_samples.strings("negative_tweets.json")
all_tweets = positive_tweets + negative_tweets

df = pd.DataFrame({"text": all_tweets})
print("Total tweets loaded:", len(df))

# ----------------------------------------------------------
# 3. Clean the text
# ----------------------------------------------------------
def clean_text(text):
    text = re.sub(r"http\S+|www\S+", "", text)      # remove URLs
    text = re.sub(r"@\w+", "", text)                # remove @mentions
    text = re.sub(r"#", "", text)                    # remove hashtag symbol (keep word)
    text = re.sub(r"[^A-Za-z\s]", "", text)          # remove punctuation/numbers
    text = text.lower().strip()
    return text

df["clean_text"] = df["text"].apply(clean_text)

# ----------------------------------------------------------
# 4. Sentiment scoring with VADER
# ----------------------------------------------------------
sia = SentimentIntensityAnalyzer()

df["compound"] = df["clean_text"].apply(lambda t: sia.polarity_scores(t)["compound"])


def label_sentiment(score):
    if score >= 0.05:
        return "Positive"
    elif score <= -0.05:
        return "Negative"
    else:
        return "Neutral"


df["sentiment"] = df["compound"].apply(label_sentiment)

# ----------------------------------------------------------
# 5. Visualize sentiment distribution
# ----------------------------------------------------------
plt.figure(figsize=(6, 5))
sns.countplot(data=df, x="sentiment", hue="sentiment",
              order=["Positive", "Neutral", "Negative"],
              palette="Set2", legend=False)
plt.title("Sentiment Distribution of Sample Social Media Posts")
plt.xlabel("Sentiment")
plt.ylabel("Number of Posts")
plt.tight_layout()
plt.savefig("sentiment_distribution.png", dpi=150)
plt.show()

# ----------------------------------------------------------
# 6. Compound score distribution
# ----------------------------------------------------------
plt.figure(figsize=(8, 5))
sns.histplot(df["compound"], bins=30, kde=True, color="purple")
plt.title("Distribution of Sentiment Compound Scores")
plt.xlabel("VADER Compound Score (-1 = very negative, +1 = very positive)")
plt.tight_layout()
plt.savefig("compound_score_distribution.png", dpi=150)
plt.show()

# ----------------------------------------------------------
# 7. Most common words in positive vs negative posts
# ----------------------------------------------------------
stop_words = set(stopwords.words("english"))


def top_words(text_series, n=15):
    words = " ".join(text_series).split()
    words = [w for w in words if w not in stop_words and len(w) > 2]
    return pd.Series(words).value_counts().head(n)


pos_words = top_words(df[df["sentiment"] == "Positive"]["clean_text"])
neg_words = top_words(df[df["sentiment"] == "Negative"]["clean_text"])

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
sns.barplot(x=pos_words.values, y=pos_words.index, ax=axes[0], hue=pos_words.index,
            palette="Greens_r", legend=False)
axes[0].set_title("Top Words - Positive Posts")

sns.barplot(x=neg_words.values, y=neg_words.index, ax=axes[1], hue=neg_words.index,
            palette="Reds_r", legend=False)
axes[1].set_title("Top Words - Negative Posts")

plt.tight_layout()
plt.savefig("top_words_by_sentiment.png", dpi=150)
plt.show()

# ----------------------------------------------------------
# 8. Summary
# ----------------------------------------------------------
print("\n--- Sentiment Summary ---")
print(df["sentiment"].value_counts())
print("\nAverage compound score:", df["compound"].mean())
