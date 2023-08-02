"""
Module creates a neural network for the Summer 2023 Machine Learning course at
the University of Texas at Dallas

@author Alexis Tudor
"""
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

class KMeansTweets:
    """
    KMeansTweets is a class that performs K-Means clustering for the Health
    Tweets Dataset:
    https://archive.ics.uci.edu/dataset/438/health+news+in+twitter
    Data hosted: https://personal.utdallas.edu/~art150530/wsjhealth.txt
    """
    def __init__(self, location, k=50, max_epochs=100):
        self.epochs = max_epochs
        self.test_progress = []
        self._data = pd.DataFrame()
        self.load_data(location)
        self.preprocess_data()
        self.centers = []
        for i in range(k):
            self.centers.append(self._data[i])


    def load_data(self, location):
        """
        This method loads the data from a link into a pandas dataframe
        """
        self._data = pd.read_csv(location, skiprows=1, index_col=False, sep="|",
                         names=["ID", "Datetime", "Tweet"],
                         usecols=["Tweet"])

    def preprocess_data(self):
        """
        This method pre-processes a pandas dataframe in order to remove
        URLs, @ mentions, and hashtags.
        """
        tweets = []
        for iter, row in self._data.iterrows():
            new_tweet = row['Tweet'].split()
            for index, word in enumerate(new_tweet):
                new_word = word.lower()
                if '@' in new_word or 'http' in new_word or 'www' in new_word or 'rt' == new_word:
                    new_word = ''
                if '#' in new_word:
                    new_word = new_tweet[index].replace('#', '')
                if '&amp' in new_word:
                    new_word = new_tweet[index].replace('&amp', ' and ')
                new_tweet[index] = new_word
            if len(new_tweet) > 0:
                tweets.append(" ".join(new_tweet))
        self._data = tweets

    def train(self):
        """
        Train the K-Means Cluster
        """
        for i in range(self.epochs):
            cluster = self.cluster()
            new_centers = []
            for j in range(len(self.centers)):
                new_centers.append(self.new_center(cluster[j]))
            self.test_progress.append(self.test())
            if new_centers == self.centers:
                break
            self.centers = new_centers

    def test(self):
        """
        Test the K-Means Cluster
        """
        clusters = self.cluster()
        averages = []
        for i in range(len(self.centers)):
            averages.append(self.average_distance(self.centers[i], clusters[i]))
        return (sum(averages) / len(averages)),

    @staticmethod
    def tweet_distance(tweet1, tweet2):
        # Union is all words in either
        # Intersection is words in both
        # Dist(A,B) = 1 - (A INT B)/(A U B)
        tweet1 = tweet1.split()
        tweet2 = tweet2.split()
        intersection = 0
        for word in tweet1:
            intersection = intersection + tweet2.count(word)
        union = len(tweet1) + len(tweet2) - intersection
        distance = 1 - (intersection/union)
        return distance

    def new_center(self, tweets):
        center_dist = 2
        center = ""
        for tweet in tweets:
            average = self.average_distance(tweet, tweets)
            if average < center_dist:
                center_dist = average
                center = tweet
        return center

    def average_distance(self, tweet, tweets):
        average = 0
        for dist_tweet in tweets:
            average = average + self.tweet_distance(tweet, dist_tweet)
        average = average / len(tweets)
        return average

    def assign(self, tweet):
        dist = 2
        center_loc = 0
        for center in self.centers:
            td = self.tweet_distance(center, tweet)
            if td < dist:
                dist = td
                center_loc = self.centers.index(center)
        return center_loc

    def cluster(self):
        cluster = []
        for _ in range(len(self.centers)):
            cluster.append([])
        for tweet in self._data:
            cluster[self.assign(tweet)].append(tweet)
        return cluster

    def plot_training_error(self):
        x_axis = []
        for i in range(len(self.test_progress)):
            x_axis.append(i)
        plt.plot(x_axis, self.test_progress)
        plt.ylabel('Error')
        plt.show()

    def sse(self):
        clusters = self.cluster()
        sse = 0
        for i in range(len(clusters)):
            for tweet1 in clusters[i]:
                sse = sse + self.tweet_distance(tweet1, self.centers[i])**2
        return sse