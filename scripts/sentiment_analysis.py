# This is a code that takes a certain dataset e.g. hotel reviews and runs it through a sentiment analysis library called VaderSentiment
# AUTHOR: Faisal Altassan

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd

# analyze the scentences
analyzer = SentimentIntensityAnalyzer()

def classify_sentiment(compound):
    if 0.9 <= compound <= 1:
        return 'Great'
    elif 0.7 <= compound <= 0.899999999999:
        return 'Very Good'
    elif 0.3 <= compound <= 0.6999999999999:
        return 'Good'
    elif 0 <= compound <= 0.29999999999:
        return 'Neutral'
    elif -0.4 <= compound <= -0.01999999999:
        return 'Bad'
    elif -0.89 <= compound <= -0.4199999999:
        return 'Very Bad'
    elif -1 <= compound <= -0.999999999:
        return 'Extremely Bad'
    else:
        return 'out of range'

def getSentimentScore(sentiment_class):
    if sentiment_class == 'Great':
        return 6
    elif sentiment_class =='Very Good':
        return 5
    elif sentiment_class == 'Good':
        return 4
    elif sentiment_class == 'Neutral':
        return 3
    elif sentiment_class == 'Bad':
        return 2
    elif sentiment_class == 'Very Bad':
        return 1
    elif sentiment_class == 'Extremely Bad':
        return 0
    else:
        return 'N/A'

df = pd.read_csv('input/sentiment-palintir-schema.csv') # change the filename to your .csv file name and location

# add sentiment class and sentiment scores columns
sentiments = []
compound_scores = []
positive_scores = []
neutral_scores = []
negative_scores = []
    
for index, row in df.iterrows():
    sentence = row['text']
    score = analyzer.polarity_scores(sentence)
    compound_score = score['compound']
    positive_score = score['pos']
    neutral_score = score['neu']
    negative_score = score['neg']
    sentiment_class = classify_sentiment(compound_score)
    sentiment_score = compound_score * 5

    compound_scores.append(compound_score)
    positive_scores.append(positive_score)
    neutral_scores.append(neutral_score)
    negative_scores.append(negative_score)
    sentiments.append(sentiment_class)

# Add the sentiment results as a new column in the DataFrame
df['sentiment'] = sentiments
df['compound_score'] = compound_scores
df['positive_score'] = positive_scores
df['neutral_score'] = neutral_scores
df['negative_score'] = negative_scores
df['sentiment_score'] = df['sentiment'].apply(lambda x: getSentimentScore(x))

# Save the updated DataFrame to a new CSV file
df.to_csv('input/topic_sentiment.csv', index=False)

print(f"Sentiment analysis completed and results saved to {'input/topic_sentiment.csv'}.")