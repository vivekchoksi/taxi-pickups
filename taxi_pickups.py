import sys
from abc import ABCMeta, abstractmethod
from sklearn.metrics import mean_squared_error
import MySQLdb
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

        :param test_example: Tuple of the form (pickup_datetime, pickup_lat, pickup_long), where
                    pickup_datetime is a datetime object, and
                    pickup_[lat,long] are floats.

        :return: Predicted number of pickups for the test example.
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
    '''
    Usage:
        dataset = Dataset(0.7, 20) # 14 examples in train set, 6 in test set
        while dataset.hasMoreTrainExamples():
            train_examples = dataset.getTrainExamples(batch_size=2)
            # Do something with the training examples...

        while dataset.hasMoreTestExamples():
            test_example = dataset.getTestExample()
            # Do something with the test example...

    TODO: Fix awkwardness when batch_size is not a divisor of the number of train
    examples.
    '''

    DATA_TABLE_NAME = "trip_data"

    def __init__(self, train_fraction, dataset_size):
        self.db = MySQLdb.connect(host="localhost", user="root", passwd="",  db="taxi_pickups")

        # The id of the last examples in the train and test set, respectively.
        self.last_train_id = int(train_fraction * dataset_size)
        self.last_test_id = dataset_size

        # The id of the next example to be fetched.
        self.current_example_id = 1

    def hasMoreTrainExamples(self):
        return self.current_example_id <= self.last_train_id

    def hasMoreTestExamples(self):
        return self.current_example_id <= self.last_test_id

    def getTrainExamples(self, batch_size):
        '''
        :param batch_size: number of training examples to return
        :return: training examples , represented as a list of tuples
        '''
        if self.current_example_id + batch_size - 1 > self.last_train_id:
            raise Exception("Cannot access example %d: outside specified " \
                            "train set range." \
                            % (self.current_example_id + batch_size - 1))

        examples = self._getExamples(self.current_example_id, num_examples=batch_size)
        self.current_example_id += batch_size
        return examples

    def getTestExample(self):
        '''
        :return: test example, represented as a tuple.
        '''
        if self.current_example_id > self.last_test_id:
            raise Exception("Cannot access example %d: outside specified " \
                            "dataset size range of %d." \
                            % (self.current_example_id, self.last_test_id))

        if self.current_example_id <= self.last_train_id:
            self.current_example_id = self.last_train_id + 1

        example = self._getExamples(self.current_example_id, num_examples=1)[0]
        self.current_example_id += 1
        return example

    def _getExamples(self, start_id, num_examples=1):
        '''
        :param start_id: id of first row to fetch
        :param num_examples: number of examples to return
        :return: examples (i.e. rows) from the data table represented as a list
                    of tuples.
        '''
        cursor = self.db.cursor()
        end_id = start_id + num_examples - 1

        query_string = "SELECT * FROM %s WHERE " \
                        "id BETWEEN %d AND %d" \
                        % (self.DATA_TABLE_NAME, start_id, end_id)

        cursor.execute(query_string)
        self.db.commit()
        return cursor.fetchall()


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

        # Generate a predicted number of pickups for every example in the test data.
        predicted_num_pickups = []
        for test_example in test_data:
            predicted_num_pickups.append(self.model.predict(test_example))

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
