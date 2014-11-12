import sys
from abc import ABCMeta, abstractmethod
from sklearn.metrics import mean_squared_error
from math import sqrt

import baseline

# Interface for our learning models.
class Model(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def train(self, dataset):
        '''
        Trains the learning model on the list of training examples provided in
        the dataset.
        '''
        pass

    @abstractmethod
    def predict(self, test_example):
        '''
        Predicts the number of pickups for the test example provided.

        :param test_example: List of tuples of the form (pickup_datetime, pickup_lat, pickup_long), where
                    pickup_datetime is a datetime object, and
                    pickup_[lat,long] are floats.

        :return: List of the predicted number of pickups, arranged in the same order as test_data.
        '''
        pass

    @abstractmethod
    def generateTestData(self):
        '''
        Generates the data we use to evaluate our learning models.

        :return: (test_data, true_num_pickups), where
                    test_data is a list of tuples of the form (pickup_datetime, pickup_lat, pickup_long), and
                    true_num_pickups is a list of the actual number of pickups observed
                        (to be used to evaluate the predictions).
        '''
        pass

# The `Dataset` class interfaces with the data.
class Dataset(object):

    def __init__(self, train_fraction, dataset_size):
        self.train_fraction = train_fraction
        self.dataset_size = dataset_size

    def getTrainExample(self, batch_size):
        '''
        :param batch_size: number of training examples to return
        :return: list of training examples
        '''
        pass

    def getTestExample(self):
        pass

# The `Evaluator` class evaluates a trained model.
class Evaluator(object):

    def __init__(self, model, dataset):
        self.model = model
        self.dataset = dataset

    def evaluate(self):
        # Get testing examples from dataset
        # Feed them to model and predict num pickups
        # Keep track of necessary statistics and metrics
        # Print statistics and performance

        # Test the model.
        test_data, true_num_pickups = self.model.generateTestData()
        predicted_num_pickups = self.model.predict(test_data)

        # Evaluate the predictions.
        self.evaluatePredictions(true_num_pickups, predicted_num_pickups)

    def evaluatePredictions(self, true_num_pickups, predicted_num_pickups):
        '''
        Prints some metrics on how well the model performed, including the RMSD.

        :param predicted_num_pickups: List of predicted num_pickups.
        :param true_num_pickups: List of observed num_pickups.

        '''
        assert(len(true_num_pickups) == len(predicted_num_pickups))

        print 'True number of pickups:\t\t' + str(true_num_pickups)
        print 'Predicted number of pickups:\t' + str(predicted_num_pickups)

        # Compute the RMSD
        rms = sqrt(mean_squared_error(true_num_pickups, predicted_num_pickups))
        print 'RMSD: %f' % rms


def getModel(modelName):
    if modelName == 'baseline':
        return baseline.Baseline()
    raise Exception("No model with name %s" % modelName)

def main():
    args = sys.argv
    if len(args) < 2:
        print 'Usage: taxi_pickups model'
        exit(1)

    # Instantiate the specified learning model.
    model = getModel(args[1])
    dataset = Dataset(0.7, 20)
    evaluator = Evaluator(model, dataset)

    # Train the model.
    model.train(dataset)

    # Evaluate the model on data from the test set.
    evaluator.evaluate()


if __name__ == '__main__':
    main()