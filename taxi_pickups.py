import sys
from abc import ABCMeta, abstractmethod
from sklearn.metrics import mean_squared_error
from math import sqrt

import baseline

# Interface for our learning models.
class Model(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def train(self):
        '''
        Trains the learning model on the list of training examples provided.
        '''
        pass

    @abstractmethod
    def test(self, test_data):
        '''
        Predicts the number of pickups for each test example provided.

        :param test_data: List of tuples of the form (pickup_datetime, pickup_lat, pickup_long), where
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

def evaluatePredictions(true_num_pickups, predicted_num_pickups):
    '''
    Prints some metrics on how well the model performed, including the RMSD.

    :param predicted_num_pickups: List of predicted num_pickups.
    :param true_num_pickups: List of observed num_pickups.

    '''
    assert(len(true_num_pickups) == len(predicted_num_pickups))

    # Compute the RMSD
    rms = sqrt(mean_squared_error(true_num_pickups, predicted_num_pickups))
    print 'RMSD: %f' % rms

def main():
    args = sys.argv
    if len(args) < 2:
        print 'Usage: taxi_pickups model'
        exit(1)

    # Instantiate the specified learning model.
    if args[1] == 'baseline':
        model = baseline.Baseline()

    # Train the model.
    model.train()

    # Test the model.
    test_data, true_num_pickups = model.generateTestData()
    predicted_num_pickups = model.test(test_data)

    # Evaluate the predictions.
    evaluatePredictions(true_num_pickups, predicted_num_pickups)

if __name__ == '__main__':
    main()