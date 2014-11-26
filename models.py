#!/usr/bin/python
import MySQLdb
import datetime
import operator
import os, sys
import numpy as np
from sklearn import linear_model, preprocessing
from abc import ABCMeta, abstractmethod
from const import Const
from feature_extractor import getFeatureVectors, getFeatureNameIndices
import util

# Interface for our learning models.
class Model(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def train(self):
        '''
        Trains the learning model on the list of training examples provided in
        the dataset (passed in through the constructor).
        '''
        pass

    @abstractmethod
    def predict(self, test_example):
        '''
        Predicts the number of pickups for the test example provided.

        :param test_example: dict mapping feature names to feature values

        :return: Predicted number of pickups for the test example.
        '''
        pass

class LinearRegression(Model):

    def __init__(self, database, dataset):
        self.db = database
        self.dataset = dataset
        self.table_name = Const.AGGREGATED_PICKUPS
        self.scaler = preprocessing.StandardScaler(with_mean=False)
        self.regressor = linear_model.SGDRegressor(
            learning_rate='constant', eta0=0.0001,
            verbose=1
        )

    def train(self):
        '''
        See Model for comments on the parameters and return value.
        '''
        row_dicts = []
        while self.dataset.hasMoreTrainExamples():
            # Get a batch of training examples, represented as a list of dicts.
            row_dicts.extend(self.dataset.getTrainExamples(Const.TRAIN_BATCH_SIZE))

        # Transform the training data into "vectorized" form.
        X = getFeatureVectors(row_dicts)

        # Get the labels of the training examples.
        y = np.array([ex['num_pickups'] for ex in row_dicts])

        print 'Memory Footprint in bytes:'
        print 'Feature dicts: ', sys.getsizeof(row_dicts)
        print 'X: ', X.data.nbytes

        # TODO: How can we get scaling to work with partial_fit?
        self.scaler.fit_transform(X, y)

        # self.regressor.partial_fit(X, y)
        self.regressor.fit(X, y)
        self.printMostPredictiveFeatures(15)

    def printMostPredictiveFeatures(self, n):
        '''
        Prints the n features whose coefficients are the highest, and the n features
        whose coefficients are the lowest.

        :param n: number of the best/worst features to print (prints 2n features total)
        '''
        feature_weights = []
        for feature_name, index in getFeatureNameIndices().iteritems():
            feature_weights.append((feature_name, self.regressor.coef_[index]))
        feature_weights.sort(key=operator.itemgetter(1))

        def printFeatureWeight(feature_weight):
            print '%s:\t%f' % (feature_weight[0], feature_weight[1])

        print ('Feature\t\tWeight')
        [printFeatureWeight(feature_weight) for feature_weight in feature_weights[:n]]
        [printFeatureWeight(feature_weight) for feature_weight in feature_weights[-n:]]

    def predict(self, test_example):
        '''
        Predicts the number of pickups at the specified time and location,
        within a 1 hour interval and 0.01 x 0.01 degrees lat/long box.

        See Model for comments on the parameters and return value.
        '''
        vectorized_example = getFeatureVectors([test_example], is_test=True)
        y = self.regressor.predict(vectorized_example)[0]
        return y

# Predicts taxi pickups by averaging past aggregated pickup
# data in the same zone and at the same hour of day.
class BetterBaseline(Model):

    def __init__(self, database, dataset):
        self.db = database
        self.dataset = dataset
        self.table_name = Const.AGGREGATED_PICKUPS

    def train(self):
        '''
        The SQL script to generate the aggregated pickups table is commented out
        because we only need to run it once.

        See Model for comments on the parameters and return value.
        '''
        # Note: this line of code isn't tested yet.
        # os.system('mysql -u root < pickups-aggregated.sql')
        pass

    def predict(self, test_example):
        '''
        Predicts the number of pickups at the specified time and location,
        within a 1 hour interval and 0.01 x 0.01 degrees lat/long box.

        See Model for comments on the parameters and return value.
        '''
        num_pickups = 0.0
        pickup_time = test_example['start_datetime']
        example_id, zone_id = test_example['id'], test_example['zone_id']
        query_string = ("SELECT AVG(num_pickups) as avg_num_pickups FROM %s "
                        "WHERE HOUR(start_datetime) = %d AND zone_id = %d AND "
                        # Hacky way to limit ourself to looking at training data.
                        "id <= %d") % \
                        (self.table_name, pickup_time.hour, zone_id, 
                        self.dataset.last_train_id)
        # print "Querying 'trip_data': " + query_string
        results = self.db.execute_query(query_string, fetch_all=False)
        if len(results) == 1:
            num_pickups = float(results[0]['avg_num_pickups'])

        return num_pickups

# Predicts taxi pickups by averaging past aggregated pickup
# data in the same zone.
class Baseline(Model):

    def __init__(self, database, dataset):
        self.db = database
        self.dataset = dataset
        self.table_name = Const.AGGREGATED_PICKUPS

    def train(self):
        '''
        The SQL script to generate the aggregated pickups table is commented out
        because we only need to run it once.

        See Model for comments on the parameters and return value.
        '''
        # Note: this line of code isn't tested yet.
        # os.system('mysql -u root < pickups-aggregated.sql')
        pass

    def predict(self, test_example):
        '''
        Predicts the number of pickups at the specified time and location, 
        within a 1 hour interval and 0.01 x 0.01 degrees lat/long box.

        See Model for comments on the parameters and return value.
        '''
        num_pickups = 0.0
        pickup_time = test_example['start_datetime']
        example_id, zone_id = test_example['id'], test_example['zone_id']
        query_string = ("SELECT AVG(num_pickups) as avg_num_pickups FROM %s "
                        "WHERE zone_id = %d AND "
                        # Hacky way to limit ourself to looking at training data.
                        "id <= %d") % \
                        (self.table_name, zone_id, self.dataset.last_train_id)
        # print "Querying 'trip_data': " + query_string
        results = self.db.execute_query(query_string, fetch_all=False)
        if len(results) == 1:
            num_pickups = float(results[0]['avg_num_pickups'])

        return num_pickups