"""
Module creates a neural network for the Summer 2023 Machine Learning course at
the University of Texas at Dallas

@author Alexis Tudor
"""
import pandas as pd
import numpy as np

class KMeansTweets:
    """
    KMeansTweets is a class that performs K-Means clustering for the Health
    Tweets Dataset:
    https://archive.ics.uci.edu/dataset/438/health+news+in+twitter
    Data hosted: https://personal.utdallas.edu/~art150530/wsjhealth.txt
    """
    def __init__(self, location):
        self._data = pd.DataFrame()
        self.load_data(location)
        self.preprocess_data()


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
        for iter, row in self._data.iterrows():
            new_tweet = row['Tweet'].split()
            for index, word in enumerate(new_tweet):
                if '@' in word or 'http' in word or 'www' in word:
                    new_tweet[index] = ''
                if '#' in word:
                    new_tweet[index] = new_tweet[index].replace('#', '')
                new_tweet[index] = new_tweet[index].lower()
            self._data[iter] = " ".join(new_tweet)
            print(self._data[iter])

    def train(self):
        """
        Train the K-Means Cluster
        """
        print("Train")

    def test(self):
        """
        Test the K-Means Cluster
        """
        print("Test")

    def update(self):
        """
        Function that updates the cluster centers
        """
        print("Update")

    def predict(self, data):
        """
        For a single instance row of data, uses the model to predict it.
        :param data: a row of the data
        :return: prediction, 1 or 0
        """
        print("Predict")
