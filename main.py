import tweepy
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


class SentimentAnalyzer:
    def __init__(self, consumer_key, consumer_secret, access_token, access_token_secret):
        auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
        auth.set_access_token(access_token, access_token_secret)

        self.api = tweepy.API(auth)
        self.stopwords = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

    def scrape_tweets(self, keyword, count):
        tweets = []
        try:
            fetched_tweets = self.api.search(q=keyword, lang="en", count=count)
            for tweet in fetched_tweets:
                tweets.append(tweet.text)
        except tweepy.TweepError as e:
            print("Error : " + str(e))

        return tweets

    def preprocess_text(self, text):
        text = re.sub(r'http\S+', '', text)  # remove URLs
        text = re.sub(r'\W', ' ', text)  # remove non-alphanumeric characters
        text = re.sub(r'\s+', ' ', text)  # remove extra whitespaces
        text = text.lower()  # convert to lowercase

        words = word_tokenize(text)  # tokenize text into words

        filtered_words = [self.lemmatizer.lemmatize(word) for word in words if word not in self.stopwords]
        preprocessed_text = ' '.join(filtered_words)

        return preprocessed_text

    def get_sentiment(self, text, sentiments):
        tfidf_vectorizer = TfidfVectorizer()
        tfidf_matrix = tfidf_vectorizer.fit_transform(text)

        X_train, X_test, y_train, y_test = train_test_split(tfidf_matrix, sentiments, test_size=0.2, random_state=42)

        model = MultinomialNB()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)

        return accuracy, report

    def analyze_sentiment(self, keyword, count):
        tweets = self.scrape_tweets(keyword, count)

        preprocessed_tweets = [self.preprocess_text(tweet) for tweet in tweets]
        sentiments = ["positive" if "positive" in tweet else "negative" for tweet in tweets]

        accuracy, report = self.get_sentiment(preprocessed_tweets, sentiments)

        print("Sentiment Analysis Results:")
        print("Accuracy:", accuracy)
        print("Classification Report:")
        print(report)

        sentiments_df = pd.DataFrame({'Tweets': tweets, 'Sentiments': sentiments})
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

    sentimentAnalyzer = SentimentAnalyzer(consumer_key, consumer_secret, access_token, access_token_secret)

    keyword = input("Enter a keyword to analyze: ")
    count = int(input("Enter the number of tweets to scrape: "))

    sentiments = sentimentAnalyzer.analyze_sentiment(keyword, count)
    sentimentAnalyzer.visualize_sentiment(sentiments)