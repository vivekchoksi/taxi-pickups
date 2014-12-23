#!/usr/bin/python
'''
The Model class and its various subclasses define a number of different
regression models, including two baseline models, linear regression, decision
tree regression, neural network regression, and support vector regression,
as well as 'auto-tuning' grid-search versions of some of the aforementioned
models. sklearn grid_search is used to determine optimal model parameters.
'''
import sys
import operator
from abc import ABCMeta, abstractmethod
from sklearn import linear_model, svm, tree, grid_search
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.datasets import SupervisedDataSet
import util
from time import time
import numpy as np
from const import Const
from feature_extractor import FeatureExtractor

# Interface for the learning models.
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

# RegressionModel perform training and testing on the regression
# model input into its constructor.
class RegressionModel(Model):
    __metaclass__ = ABCMeta

    def __init__(self, database, dataset, regressor_model, sparse=True):
        self.db = database
        self.dataset = dataset
        self.table_name = Const.AGGREGATED_PICKUPS
        self.regressor = regressor_model

        # `sparse` determines whether data should be represented as sparse scipy
        # matrices as opposed to dense ones. (Some models such as the decision tree
        # regression model require a dense representation.)
        self.feature_extractor = FeatureExtractor(sparse)

        util.verbosePrint(self.regressor)

    def train(self):
        # Populate `row_dicts` with all training examples, represented as a
        # list of dicts.
        row_dicts = []
        while self.dataset.hasMoreTrainExamples():
            row_dicts.extend(self.dataset.getTrainExamples(Const.TRAIN_BATCH_SIZE))

        # Transform the training data into "vectorized" form.
        X = self.feature_extractor.getFeatureVectors(row_dicts)
        # Get the labels of the training examples.
        y = np.array([train_example['num_pickups'] for train_example in row_dicts])

        self.regressor.fit(X, y)

        if util.VERBOSE:
            self._printMemoryStats(row_dicts, X)
            self._printFeatureWeights(n=15)
            if isinstance(self.regressor, tree.DecisionTreeRegressor):
                self._exportToDotfile()

    def predict(self, test_example):
        vectorized_example = self.feature_extractor.getFeatureVectors(
            [test_example], is_test=True)
        y = self.regressor.predict(vectorized_example)[0]

        # Set 0 to be the lower bound for predictions, since negative numbers
        # of taxi pickups are impossible.
        y = max(0.0, y)
        return y

    def _printMemoryStats(self, row_dicts, X):
        print '\n\t---- Memory usage stats ----'
        print '\tTraining feature dicts: \t', sys.getsizeof(row_dicts), ' bytes used'
        if hasattr(X.data, 'nbytes'):
            print '\tVectorized training data: \t', X.data.nbytes, ' bytes used\n'
        else:
            print '\tVectorized training data: \t', sys.getsizeof(X), ' bytes used\n'

    def _printFeatureWeights(self, n=None):
        '''
        Print the model's features and their corresponding weights, if the
        model has feature coefficients.

        :param n: print this many of the best/worst features (2n features total).
                  If n is None, print all features.
        '''
        feature_weights_dict = self.getFeatureWeights()
        if feature_weights_dict is None:
            return

        feature_weights = [(feature_name, weight) for feature_name, weight
                           in feature_weights_dict.iteritems()]
        feature_weights.sort(key=operator.itemgetter(1))

        def printFeatureWeight(feature_weight):
            print '%s\t%f' % (feature_weight[0], feature_weight[1])

        print 'Feature\t\tWeight'

        if n is None:
            [printFeatureWeight(feature_weight) for feature_weight in feature_weights]
        else:
            [printFeatureWeight(feature_weight) for feature_weight in feature_weights[:n]]
            [printFeatureWeight(feature_weight) for feature_weight in
             feature_weights[-min(n, len(feature_weights) - n):]]

    def getFeatureWeights(self, zone_id=None):
        '''
        Return the model's features and their corresponding weights, if the model has feature coefficients.

        :param zone_id: prints features only relevant to this zone.
            If zone_id is None, considers all features.
            Note: the implementation is hacky. It considers a feature relevant to a given zone_id if the zone_id
            appears anywhere in the feature's name.
        :return dictionary mapping feature names to weights.
        '''
        if not hasattr(self.regressor, 'coef_'):
            print '\tCannot get feature weights for the model.'
            return None

        feature_weights = {}
        for feature_name, index in self.feature_extractor.getFeatureNameIndices().iteritems():
            if zone_id is None or str(zone_id) in feature_name:
                feature_weights[feature_name] = self.regressor.coef_[index]
        return feature_weights

    def _exportToDotfile(self):
        '''
        Assumes that self.regressor is a decision tree regressor model and
        exports the trained decision tree to a .dot file.
        '''
        out_file_prefix = Const.OUTFILE_DIRECTORY + 'dtr_' + str(int(time()))
        out_file_dot = out_file_prefix + '.dot'
        out_file_png = out_file_prefix + '.png'
        print '\n\tExporting decision tree to dotfile: ' + out_file_dot
        print '\tTo view the graph, first convert to png using the ' \
            'command: dot -Tpng %s -o %s' % (out_file_dot, out_file_png)
        print '\tNote: This command requires the program graphviz, which ' \
            'is installed on the corn machines.'
        tree.export_graphviz(self.regressor, out_file=out_file_dot)


# Neural network regression model using the PyBrain library.
class NeuralNetworkRegression(RegressionModel):
    __metaclass__ = ABCMeta

    def __init__(self, database, dataset, hidden_layer_multiplier):
        nnr = NeuralNetworkRegressor(hidden_layer_multiplier)
        RegressionModel.__init__(self, database, dataset, nnr, sparse=False)

    def __str__(self):
        return 'nnr [neural network model]'

class NeuralNetworkRegressor:
    '''
    This is a wrapper around PyBrain's neural network library. It
    implements an API that mimics the interface for sklearn's regressors.
    '''

    def __init__(self, hidden_layer_mutliplier):
        self.hidden_layer_multiplier = hidden_layer_mutliplier

    def fit(self, X, y):
        '''
        Train the model.

        :param X: list of numpy arrays representing the training samples.
        :param y: numpy array representing the training samples' true values.
        '''
        input_dimension = len(X[0])
        self.nnw = buildNetwork(input_dimension, input_dimension * self.hidden_layer_multiplier, 1)

        if util.VERBOSE:
            print 'Generating neural network data set:'
            print '\tInput layer dimension: %d' % input_dimension
            print '\tHidden layer dimension: %d ' % (input_dimension * self.hidden_layer_multiplier)

        # Create a data set for training samples with input_demnsion number of features, and outputs with dimension 1.
        data_set = SupervisedDataSet(input_dimension, 1)
        for i in xrange(len(y)):
            data_set.appendLinked(X[i], np.array(y[i]))

        if util.VERBOSE:
            print 'Finished generating neural network data set.'
            print 'Starting to train neural network.'

        # Train the neural network with backpropagation on the data set.
        self.trainer = BackpropTrainer(self.nnw, dataset=data_set)

        for i in xrange(3):
            error = self.trainer.train()
            print 'Iteration: %d\tError: %f' % (i, error)

        if util.VERBOSE:
            print 'Finished training neural network.'

    def predict(self, x):
        '''
        Predict the output for sample x.

        :param x: an array of one element, which is a numpy array defining one sample.
        :return array(y), where array is a numpy array
        '''
        return self.nnw.activate(x[0])


# Regression model whose hyper-parameters are automatically tuned on training
# data using grid search and cross-validation.
class AutoTunedRegressionModel(RegressionModel):
    __metaclass__ = ABCMeta

    def __init__(self, database, dataset, regressor_model, params, n_jobs=4, cv=3, sparse=True):
        sgd_regressor = grid_search.GridSearchCV(
            regressor_model,
            params, 
            n_jobs=n_jobs,
            refit=True,
            cv=cv,
            verbose=1 if util.VERBOSE else 0
        )
        RegressionModel.__init__(self, database, dataset, sgd_regressor, sparse)

    def train(self):
        super(AutoTunedRegressionModel, self).train()
        if util.VERBOSE:
            print "\nOptimal Parameters: %s" % self.get_params()

    def get_params(self):
        return self.regressor.best_params_

class LinearRegression(RegressionModel):
    def __init__(self, database, dataset):
        # The parameters used here are the best of the values tested
        # using grid-search.
        sgd_regressor = linear_model.SGDRegressor(
            n_iter=8000, # Takes many iterations to converge.
            alpha=0.0, # Works better without regularization.
            learning_rate='invscaling',
            eta0=0.2, # Converges faster with higher-than-default initial learning rate.
            power_t=0.4,
            verbose=1 if util.VERBOSE else 0
        )
        RegressionModel.__init__(self, database, dataset, sgd_regressor)

    def __str__(self):
        return 'linear [linear regression model]'

class AutoTunedLinearRegression(AutoTunedRegressionModel):

    def __init__(self, database, dataset):
        # Define the parameter values to sweep across during grid-search.
        params = {
            'n_iter': [2000, 3000, 4000],
            'alpha': [0.0],
            'learning_rate': ['invscaling'],
            'eta0': [0.1, 0.2, 0.3],
            'power_t': [0.05, 0.1, 0.2]
        }
        sgd_regressor = linear_model.SGDRegressor()
        cv = util.getCrossValidator(1, 0.9, dataset.trainingExamplesLeft)
        AutoTunedRegressionModel.__init__(self, database, dataset, sgd_regressor, params, cv=cv)

    def __str__(self):
        return 'autolinear [auto tuned linear regression model]'

class SupportVectorRegression(RegressionModel):
    def __init__(self, database, dataset):
        svr_regressor = svm.SVR(
            C=10000000.0, # With lower C values, the SVR underfits.
            verbose=util.VERBOSE
        )
        RegressionModel.__init__(self, database, dataset, svr_regressor)

    def __str__(self):
        return 'svr [support vector regression model]'

class AutoTunedSVR(AutoTunedRegressionModel):

    def __init__(self, database, dataset):
        # Define parameters to sweep across during grid-search.
        params = {
            'C': [1000, 10000, 100000, 1000000, 10000000],
            'epsilon': [0.01, 0.1, 1, 10],
            'kernel': ['linear', 'rbf', 'poly', 'sigmoid']
        }
        regressor = svm.SVR()
        cv = util.getCrossValidator(1, 0.9, dataset.trainingExamplesLeft)
        AutoTunedRegressionModel.__init__(self, database, dataset, regressor, params, cv=cv)

    def __str__(self):
        return 'autosvr [auto tuned support vector regression model]'

class DecisionTreeRegression(RegressionModel):
    def __init__(self, database, dataset):
        # NOTE: The decision tree is very sensitive to max_depth and
        # min_samples_leaf parameters. These can control the degree of over /
        # under-fitting. The parameters used here are the best of the values
        # tested using grid-search.
        dt_regressor = tree.DecisionTreeRegressor(
            max_features=0.9,
            max_depth=100,
            min_samples_leaf=2
        )
        RegressionModel.__init__(self, database, dataset, dt_regressor, sparse=False)

    def __str__(self):
        return 'dtr [decision tree regression model]'

class AutoTunedDecisionTree(AutoTunedRegressionModel):

    def __init__(self, database, dataset):
        # Define parameters to sweep across during grid-search.
        params = {
            'max_features': [0.7, 0.9, 'sqrt'],
            'max_depth': [10, 50, 100],
            'min_samples_leaf': [2, 5, 10]
        }
        dt_regressor = tree.DecisionTreeRegressor()
        cv = util.getCrossValidator(1, 0.9, dataset.trainingExamplesLeft)
        AutoTunedRegressionModel.__init__(self, database, dataset, dt_regressor, params, cv=cv, sparse=False)

    def __str__(self):
        return 'autodtr [auto tuned decision tree regression model]'

# Predict taxi pickups by averaging past aggregated pickup
# data in the same zone and at the same hour of day.
class BetterBaseline(Model):

    def __init__(self, database, dataset):
        self.db = database
        self.dataset = dataset
        self.table_name = Const.AGGREGATED_PICKUPS

    def train(self):
        pass

    def predict(self, test_example):
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
        results = self.db.execute_query(query_string, fetch_all=False)
        if len(results) == 1:
            num_pickups = float(results[0]['avg_num_pickups'])

        return num_pickups


    def __str__(self):
        return "betterbaseline [baseline version 2]"

# Predict taxi pickups by averaging past aggregated pickup
# data in the same zone.
class Baseline(Model):

    def __init__(self, database, dataset):
        self.db = database
        self.dataset = dataset
        self.table_name = Const.AGGREGATED_PICKUPS

    def train(self):
        pass

    def predict(self, test_example):
        num_pickups = 0.0
        pickup_time = test_example['start_datetime']
        example_id, zone_id = test_example['id'], test_example['zone_id']
        query_string = ('SELECT AVG(num_pickups) as avg_num_pickups FROM %s '
                        'WHERE zone_id = %d AND '
                        # Hacky way to limit ourself to looking at training data.
                        'id <= %d') % \
                        (self.table_name, zone_id, self.dataset.last_train_id)
        results = self.db.execute_query(query_string, fetch_all=False)
        if len(results) == 1:
            num_pickups = float(results[0]['avg_num_pickups'])

        return num_pickups

    def __str__(self):
        return "baseline [baseline version 1]"
