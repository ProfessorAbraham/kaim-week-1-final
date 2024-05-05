# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from textblob import TextBlob
import seaborn as sns
from datetime import datetime
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
# Load the dataset
df = pd.read_csv('../data/raw_analyst_ratings.csv')
# Ensure you have the necessary NLP libraries and datasets
nltk.download('punkt')
nltk.download('stopwords')
# Text Analysis
## Sentiment Analysis
df['sentiment'] = df['headline'].apply(lambda x: TextBlob(x).sentiment.polarity)
## Topic Modeling - Word Cloud for visualization
stop_words = set(stopwords.words('english'))
text = ' '.join(df['headline'])
wordcloud = WordCloud(stopwords=stop_words, background_color='white').generate(text)
# Display Word Cloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()
# Tokenize and vectorize headlines
vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
X = vectorizer.fit_transform(df['headline'])
# Topic modeling using Latent Dirichlet Allocation (LDA)
lda = LatentDirichletAllocation(n_components=5, random_state=42)
lda.fit(X)
# Get the top words for each topic
feature_names = vectorizer.get_feature_names_out()
topics = []
for topic_idx, topic in enumerate(lda.components_):
    top_words_idx = topic.argsort()[:-10 - 1:-1]
    top_words = [feature_names[i] for i in top_words_idx]
    topics.append(top_words)
# Display sentiment analysis results
print("Sentiment Analysis Results:")
print(df[['headline', 'sentiment']])
# Display top words for each topic
print("\nTop Words for Each Topic:")
for i, topic in enumerate(topics):
    print(f"Topic {i + 1}: {', '.join(topic)}")

    # Prepare the Data
df['date'] = pd.to_datetime(df['date'], format='ISO8601')
# Group by publication date and calculate mean sentiment
sentiment_by_date = df.groupby(df['date'].dt.date)['sentiment'].mean()
# Group by publication date and calculate mean sentiment
sentiment_by_date = df.groupby(df['date'].dt.date)['sentiment'].mean()
# Plotting temporal trends in sentiment
plt.figure(figsize=(10, 5))
plt.plot(sentiment_by_date.index, sentiment_by_date.values)
plt.title('Temporal Trends in Sentiment')
plt.xlabel('Date')
plt.ylabel('Sentiment Score (Mean)')
plt.grid(True)
plt.show()
# Analyze publication frequency over time
publication_frequency = df.groupby(df['date'].dt.date).size()
# Plotting publication frequency over time
plt.figure(figsize=(10, 5))
plt.plot(publication_frequency.index, publication_frequency.values)
plt.title('Publication Frequency Over Time')
plt.xlabel('Date')
plt.ylabel('Number of Articles')
plt.grid(True)
plt.show()

import matplotlib.pyplot as plt

# Assuming sentiment_scores is a list or array containing sentiment scores
sentiment_scores = [0.2, 0.5, -0.1, 0.8, -0.4, 0.6, 0.3, -0.2, -0.7, 0.1]

# Create a histogram of sentiment scores
plt.figure(figsize=(8, 6))
plt.hist(sentiment_scores, bins=10, color='skyblue', edgecolor='black')

# Add labels and title
plt.xlabel('Sentiment Score')
plt.ylabel('Frequency')
plt.title('Headline Sentiment Distribution')

# Add a vertical line at the mean sentiment score
plt.axvline(x=sum(sentiment_scores) / len(sentiment_scores), color='red', linestyle='--', label='Mean Sentiment Score')
plt.legend()

# Show plot
plt.grid(True)
plt.show()