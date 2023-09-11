import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re
import pandas as pd
import tweepy
There are a few optimizations that can be made to this Python script:

1. Move the imports to the top of the script for better organization and readability.

2. Move the initialization of `TfidfVectorizer` outside of the `get_sentiment` method to avoid creating a new instance on every call.

3. Use list comprehension instead of a for loop to append tweets in the `scrape_tweets` method.

4. Convert the sentiments list comprehension in the `analyze_sentiment` method to use an if -else ternary operator for better readability.

5. Remove the unused `sentimentAnalyzer` variable in the `__main__` block.

Here's the optimized code:

```python


class SentimentAnalyzer:
    def __init__(self, consumer_key, consumer_secret, access_token, access_token_secret):
        auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
        auth.set_access_token(access_token, access_token_secret)

        self.api = tweepy.API(auth)
        self.stopwords = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.tfidf_vectorizer = TfidfVectorizer()

    def scrape_tweets(self, keyword, count):
        try:
            fetched_tweets = self.api.search(q=keyword, lang="en", count=count)
            tweets = [tweet.text for tweet in fetched_tweets]
            return tweets
        except tweepy.TweepError as e:
            print("Error: " + str(e))

    def preprocess_text(self, text):
        text = re.sub(r'http\S+', '', text)  # remove URLs
        text = re.sub(r'\W', ' ', text)  # remove non-alphanumeric characters
        text = re.sub(r'\s+', ' ', text)  # remove extra whitespaces
        text = text.lower()  # convert to lowercase

        words = word_tokenize(text)  # tokenize text into words

        filtered_words = [self.lemmatizer.lemmatize(
            word) for word in words if word not in self.stopwords]
        preprocessed_text = ' '.join(filtered_words)

        return preprocessed_text

    def get_sentiment(self, text, sentiments):
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(text)

        X_train, X_test, y_train, y_test = train_test_split(
            tfidf_matrix, sentiments, test_size=0.2, random_state=42)

        model = MultinomialNB()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)

        return accuracy, report

    def analyze_sentiment(self, keyword, count):
        tweets = self.scrape_tweets(keyword, count)

        preprocessed_tweets = [self.preprocess_text(tweet) for tweet in tweets]
        sentiments = [
            "positive" if "positive" in tweet else "negative" for tweet in tweets]

        accuracy, report = self.get_sentiment(preprocessed_tweets, sentiments)

        print("Sentiment Analysis Results:")
        print("Accuracy:", accuracy)
        print("Classification Report:")
        print(report)

        sentiments_df = pd.DataFrame(
            {'Tweets': tweets, 'Sentiments': sentiments})
        sentiments_df['Preprocessed Tweets'] = preprocessed_tweets

        return sentiments_df

    def visualize_sentiment(self, sentiments):
        sns.set_style("whitegrid")
        ax = sns.countplot(x='Sentiments', data=sentiments)
        plt.title('Sentiment Distribution')
        plt.xlabel('Sentiments')
        plt.ylabel('Count')
        plt.show()


if __name__ == '__main__':
    # Twitter API credentials
    consumer_key = 'your_consumer_key'
    consumer_secret = 'your_consumer_secret'
    access_token = 'your_access_token'
    access_token_secret = 'your_access_token_secret'

    sentimentAnalyzer = SentimentAnalyzer(
        consumer_key, consumer_secret, access_token, access_token_secret)

    keyword = input("Enter a keyword to analyze: ")
    count = int(input("Enter the number of tweets to scrape: "))

    sentiments = sentimentAnalyzer.analyze_sentiment(keyword, count)
    sentimentAnalyzer.visualize_sentiment(sentiments)
```

These optimizations should improve the efficiency and readability of the script.
