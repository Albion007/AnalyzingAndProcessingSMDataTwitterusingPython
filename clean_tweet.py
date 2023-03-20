import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

tweets_df = pd.read_excel('C:\\Users\\Amorii7.AMORII7\\Desktop\\vicinitas_search_results-2.xlsx', sheet_name='tweets')

def clean_tweet(tweet):
    # Convert to lowercase
    tweet = tweet.lower()

    # Tokenize the tweet
    tokens = word_tokenize(tweet)

    # Remove stop words
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [token for token in tokens if token not in stop_words]

    # Stem the words
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(token) for token in filtered_tokens]

    # Join the stemmed tokens back into a string
    cleaned_tweet = ' '.join(stemmed_tokens)

    return cleaned_tweet

# Apply the clean_tweet function to the 'Text' column and store the cleaned tweets in a new column called 'cleaned_tweet'
tweets_df['cleaned_tweet'] = tweets_df['Text'].apply(clean_tweet)

# Print the first 5 rows of the DataFrame
print(tweets_df.head())
#print(tweets_df['cleaned_tweet'])