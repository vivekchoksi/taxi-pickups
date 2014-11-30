#!/usr/bin/python
import MySQLdb
import datetime
import sys
import util
import numpy as np
from sklearn import linear_model, preprocessing, svm, tree
from abc import ABCMeta, abstractmethod
from const import Const
from feature_extractor import getFeatureVectors

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

    @abstractmethod
    def __str__(self):
        pass

# This class can perform training and testing on the input regressor
# model. Specific model classes can subclass from `RegressionModel`.
class RegressionModel(Model):
    __metaclass__ = ABCMeta

    def __init__(self, database, dataset, regressor_model, sparse=True):
        self.db = database
        self.dataset = dataset
        self.table_name = Const.AGGREGATED_PICKUPS
        self.regressor = regressor_model

        # Whether data should be represented as sparse scipy matrices as
        # opposed to dense ones. (Some models such as the decision tree
        # regression model require a dense representation.)
        self.sparse = sparse

    def train(self):
        '''
        See Model for comments on the parameters and return value.
        We are using `fit()` rather than `partial_fit()` since the January
        data is small enough to permit fitting all data into RAM.
        '''
        # Populate `row_dicts` with all training examples, represented as a
        # list of dicts.
        row_dicts = []
        while self.dataset.hasMoreTrainExamples():
            row_dicts.extend(self.dataset.getTrainExamples(Const.TRAIN_BATCH_SIZE))

        # Transform the training data into "vectorized" form.
        X = getFeatureVectors(row_dicts, use_sparse=self.sparse)
        # Get the labels of the training examples.
        y = np.array([train_example['num_pickups'] for train_example in row_dicts])

        self.regressor.fit(X, y)

        if util.VERBOSE:
            self._printMemoryStats(row_dicts, X)
            util.printMostPredictiveFeatures(self.regressor, 15)

    def predict(self, test_example):
        '''
        Predicts the number of pickups at the specified time and location,
        within a 1 hour interval and 0.01 x 0.01 degrees lat/long box.

        See Model for comments on the parameters and return value.
        '''
        vectorized_example = getFeatureVectors([test_example], is_test=True, use_sparse=self.sparse)
        y = self.regressor.predict(vectorized_example)[0]
        return y

    def _printMemoryStats(self, row_dicts, X):
        print '\n\t---- Memory usage stats ----'
        print '\tTraining feature dicts: \t', sys.getsizeof(row_dicts), " bytes used"
        if hasattr(X.data, 'nbytes'):
            print '\tVectorized training data: \t', X.data.nbytes, " bytes used\n"


class LinearRegression(RegressionModel):
    def __init__(self, database, dataset):
        sgd_regressor = linear_model.SGDRegressor(
            n_iter=15,
            verbose=1 if util.VERBOSE else 0
        )
        RegressionModel.__init__(self, database, dataset, sgd_regressor)

    def __str__(self):
        return 'linear [linear regression model]'


class SupportVectorRegression(RegressionModel):
    def __init__(self, database, dataset):
        svr_regressor = svm.SVR(
            verbose=util.VERBOSE
        )
        RegressionModel.__init__(self, database, dataset, svr_regressor)

    def __str__(self):
        return 'svr [support vector regression model]'

class DecisionTreeRegression(RegressionModel):
    def __init__(self, database, dataset):
        dt_regressor = tree.DecisionTreeRegressor()
        RegressionModel.__init__(self, database, dataset, dt_regressor, sparse=False)

    def __str__(self):
        return 'dtr [decision tree regression model]'

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
        query_string = ('SELECT AVG(num_pickups) as avg_num_pickups FROM %s '
                        'WHERE HOUR(start_datetime) = %d AND zone_id = %d AND '
                        # Hacky way to limit ourself to looking at training
                        # data. This assumes that training data is ordered
                        # by increasing id.
                        'id <= %d') % \
                        (self.table_name, pickup_time.hour, zone_id, 
                        self.dataset.last_train_id)
        util.verbosePrint('Querying `trip_data`: ' + query_string)
        results = self.db.execute_query(query_string, fetch_all=False)
        if len(results) == 1:
            num_pickups = float(results[0]['avg_num_pickups'])

        return num_pickups


    def __str__(self):
        return "baseline v2 [betterbaseline]"

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
        query_string = ('SELECT AVG(num_pickups) as avg_num_pickups FROM %s '
                        'WHERE zone_id = %d AND '
                        # Hacky way to limit ourself to looking at training data.
                        'id <= %d') % \
                        (self.table_name, zone_id, self.dataset.last_train_id)
        util.verbosePrint('Querying `trip_data`: ' + query_string)
        results = self.db.execute_query(query_string, fetch_all=False)
        if len(results) == 1:
            num_pickups = float(results[0]['avg_num_pickups'])

        return num_pickups

    def __str__(self):
        return "baseline v1 [baseline]"
