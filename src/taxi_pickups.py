#!/usr/bin/python

import os
os.environ['MPLCONFIGDIR'] = "../"
import random
from math import sqrt
from optparse import OptionParser
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
import MySQLdb
from models import *
import datetime

class Database(object):

    def __init__(self, is_local):
        self.is_local = is_local
        self._connect()

    def _connect(self):
        '''
        Connect to the MySQL database.
        '''
        if self.is_local:
            self.db = MySQLdb.connect(
                host='localhost',
                user='root',
                passwd='',
                db=Const.DATABASE_NAME)
        else:
            self.db = MySQLdb.connect(
                host=Const.MYSQL_HOST,
                user=Const.MYSQL_USER,
                passwd=Const.MYSQL_PASSWD,
                db=Const.DATABASE_NAME)

    def execute_query(self, query_string, fetch_all=True):
        '''
        :param query_string: sql query as a query_string
        :param fetch_all: True if you want all the results or false if you want
            only one result

        :return: list of rows each represented as a dict - a mapping from column
            names to values. Column names will be what you name columns in case 
            of derived columns such as avg() (using the 'as' keyword in sql)
        '''
        cursor = None
        try:
            cursor = self.db.cursor()
            cursor.execute(query_string)
        except (AttributeError, MySQLdb.OperationalError):
            # Reconnect to the database if necessary.
            self._connect()
            cursor = self.db.cursor()
            cursor.execute(query_string)

        self.db.commit()
        if fetch_all:
            tuple_results = cursor.fetchall()
        else:
            tuple_results = [cursor.fetchone()]
            if (None,) in tuple_results: # get rid of NULL result
                tuple_results.remove((None,))
            elif None in tuple_results: # in case there was no result
                tuple_results.remove(None)
        results = []
        for i, row_tuple in enumerate(tuple_results):
            results.append({
                col_tuple[0]: row_tuple[x] \
                for x, col_tuple in enumerate(cursor.description)
            })
        return results

# The `Dataset` class interfaces with the data.
class Dataset(object):
    '''
    This class assumes that the mysql table used as the dataset is sorted such
    that training examples always come before (i.e. have smaller ids) the test 
    examples in the sorted order.

    Usage:
        dataset = Dataset(0.7, 20) # 14 examples in train set, 6 in test set
        while dataset.hasMoreTrainExamples():
            train_examples = dataset.getTrainExamples(batch_size=2)
            # Do something with the training examples...

        while dataset.hasMoreTestExamples():
            test_examples = dataset.getTestExamples(Const.TEST_BATCH_SIZE)
            # Do something with the test example...
    '''

    def __init__(self, train_fraction, dataset_size, database, table_name):
        self.db = database
        self.table_name = table_name # table to read examples from
        self.trainingExamplesLeft = int(train_fraction * dataset_size)
        self.testingExamplesLeft = dataset_size - self.trainingExamplesLeft
        self.last_train_id = self._getLastTrainID()
        self.last_fetched_id = 0

    def hasMoreTrainExamples(self):
        return self.trainingExamplesLeft > 0

    def hasMoreTestExamples(self):
        return self.testingExamplesLeft > 0

    def switchToTestMode(self):
        self.last_fetched_id = self.last_train_id

    def getTrainExamples(self, batch_size=1):
        '''
        :param batch_size: number of training examples to return
        :return: training examples represented as a list of dicts. These may be
            fewer than batch_size in case there are no more training examples.
        '''
        if not self.hasMoreTrainExamples():
            raise Exception('No more training examples left.')
        if batch_size > self.trainingExamplesLeft:
            batch_size = self.trainingExamplesLeft

        examples = self._getExamples(batch_size)
        self.trainingExamplesLeft -= batch_size
        return examples

    def getTestExamples(self, batch_size=1):
        '''
        :return: test example, represented as a dict.
        '''
        if not self.hasMoreTestExamples():
            raise Exception('No more test examples left.')

        if self.last_fetched_id < self.last_train_id:
            self.switchToTestMode()

        if batch_size > self.testingExamplesLeft:
            batch_size = self.testingExamplesLeft

        examples = self._getExamples(batch_size)
        self.testingExamplesLeft -= batch_size
        return examples

    def _getExamples(self, num_examples=1):
        '''
        :param start_id: id of first row to fetch
        :param num_examples: number of examples to return
        :return: examples (i.e. rows) from the data table represented as a dicts
            that map column names to column values
        '''
        query_string = ('SELECT * FROM %s WHERE id > %d limit %d') \
                        % (self.table_name, self.last_fetched_id, num_examples)

        results = self.db.execute_query(query_string)
        self.last_fetched_id = results[len(results) - 1]['id']
        return results

    def _getLastTrainID(self):
        query_string = ('SELECT MAX(id) as max_id FROM '
                        '(SELECT id FROM %s LIMIT %d) T') \
                        % (self.table_name, self.trainingExamplesLeft)

        return self.db.execute_query(query_string, fetch_all=False)[0]['max_id']

    def __str__(self):
        info = 'Num Training Examples: %d, Num Testing Examples: %d' % \
            (self.trainingExamplesLeft, self.testingExamplesLeft)
        info += ('\nFrom table: %s' % self.table_name)
        return info

# The `Evaluator` class evaluates a trained model.
class Evaluator(object):

    def __init__(self, model, dataset, plot_error, print_feature_weights):
        self.model = model
        self.dataset = dataset
        self.plot_error = plot_error
        self.print_feature_weights = print_feature_weights

    def evaluate(self):
        '''
        Evaluate the model on a test set, and print out relevant statistics
        (e.g. RMSD).
        '''

        # Generate a predicted number of pickups for every example in the test
        # data.
        predicted_num_pickups = []
        true_num_pickups = []
        self.dataset.switchToTestMode()
        while self.dataset.hasMoreTestExamples():
            test_examples = self.dataset.getTestExamples(Const.TEST_BATCH_SIZE)
            for test_example in test_examples:
                predicted_num_pickups.append(self.model.predict(test_example))
                true_num_pickups.append(test_example['num_pickups'])

        # Print the feature weights specific to a zone.
        if self.print_feature_weights:
            start_datetime = datetime.datetime(2013, 1, 20)
            self._plotFeatureWeights(15504, start_datetime)
            self._plotFeatureWeights(14901, start_datetime)

        # Evaluate the predictions.
        self._evaluatePredictions(true_num_pickups, predicted_num_pickups)

        # Plot the prediction error by true number of pickups.
        if self.plot_error:
            self._plotPredictionError(true_num_pickups, predicted_num_pickups)
            self._plotPredictionHistogram(true_num_pickups, predicted_num_pickups)

    def _evaluatePredictions(self, true_num_pickups, predicted_num_pickups):
        '''
        Prints some metrics on how well the model performed, including the RMSD.

        :param predicted_num_pickups: List of predicted num_pickups.
        :param true_num_pickups: List of observed num_pickups.

        '''
        assert(len(true_num_pickups) == len(predicted_num_pickups))

        if util.VERBOSE:
            self._printRandomTrainingExamples(true_num_pickups, predicted_num_pickups)

        # Compute and print mean absolute error.
        m = metrics.mean_absolute_error(true_num_pickups, predicted_num_pickups)
        print 'Mean Absolute Error: %f' % m

        # Compute and print root mean squared error.
        msd = metrics.mean_squared_error(true_num_pickups, predicted_num_pickups)
        rmsd = sqrt(msd)
        print 'RMSD: %f' % rmsd

        sum_squared_errors = msd * float(len(true_num_pickups))
        mit_metric = 1.0 / (1.0 + sqrt(sum_squared_errors))
        print 'MIT metric: %f' % mit_metric

        # Compute and print coefficient of determination R^2.
        m = metrics.r2_score(true_num_pickups, predicted_num_pickups)
        print 'R^2 Score: %f' % m
        print

    def _printRandomTrainingExamples(self, true_num_pickups, predicted_num_pickups, num_examples=30):
        '''
        Prints out the true and predicted values for a small set of randomly
        selected training examples.
        '''
        print '\n\tComparison between true and predicted num pickups...'
        print '\t... for', num_examples, 'randomly selected test examples...'
        print '\tTrue value\tPredicted value'

        random_indices = random.sample(xrange(len(true_num_pickups)), num_examples) \
            if len(true_num_pickups) > num_examples \
            else xrange(len(true_num_pickups))

        for i in random_indices:
            print '\t', true_num_pickups[i], '\t\t', predicted_num_pickups[i]
        print

    def _plotFeatureWeights(self, zone_id, start_datetime, num_hours=7*24):
        '''
        :param zone_id: only use features relevant to this zone.
        :param start_datetime: datetime at which to start extracting features.
        :param num_hours: number of hours to plot

        Generates a stacked bar chart of all the features weights used to predict the number of pickups in zone zone_id
        for each hour of the week (from Sunday 12am to Saturday 11pm).
        '''
        if not hasattr(self.model, 'feature_extractor'):
            print '\tCannot plot features for the model.'
            return
        else:
            util.verbosePrint('Plotting features and their weights for each hour.')
            util.verbosePrint('\tStart time: %s' % str(start_datetime))
            util.verbosePrint('\tDuration : %s hours' % str(num_hours))
            util.verbosePrint('')

        # Mapping from all features to their corresponding weights.
        #   EX: feature_weights['Zone_HourOfDay=15402_14'] = 324.4565
        feature_weights = self.model.getFeatureWeights()

        # For each data point in the time range, get the weight for each of its features.
        # plot_values is a mapping from feature templates to a list of all their values at each time step.
        #   EX: plot_values['Zone_HourOfDay'] = [324.4565, 221.498, ... ]
        plot_values = {}

        for time_step in xrange(num_hours):
            curr_datetime = start_datetime + datetime.timedelta(hours=time_step)
            test_example = {'zone_id': zone_id, 'start_datetime': curr_datetime}

            # test_example_features is a mapping from feature templates to their identifiers
            #   EX: test_example_features['Zone_HourOfDay'] = 15402_14
            test_example_features = self.model.feature_extractor.getFeatureDict(test_example)

            for feature_template, identifier in test_example_features.iteritems():
                if feature_template not in plot_values:
                    plot_values[feature_template] = [0] * num_hours
                feature_name = '%s=%s' % (feature_template, identifier)
                if feature_name in feature_weights:
                    plot_values[feature_template][time_step] = feature_weights[feature_name]

        # Print out feature weight values, where each column represents a feature template, and each row
        # is the weights at one hour. This is useful for copy and pasting into a CSV file.
        '''
        for feature_template in plot_values.keys():
            print '%s,' % feature_template,
        print

        for time_step in xrange(num_hours):
            for feature_template in plot_values.keys():
                print '%s,' % plot_values[feature_template][time_step],
            print
        print
        '''

        # Generate stacked bar chart, whose series are the feature templates.

        # Order these feature templates first, then all the remaining feature templates in plot_values in any order.
        feature_templates = ['Zone', 'DayOfWeek', 'HourOfDay', 'Zone_DayOfWeek', 'Zone_HourOfDay']
        for feature_template in list(feature_templates):
            if feature_template not in plot_values.keys():
                feature_templates.remove(feature_template)
        for feature_template in plot_values.keys():
            if feature_template not in feature_templates:
                feature_templates.append(feature_template)

        colors = 'bgrcmy'
        indices = [i for i in xrange(num_hours)]
        series_index = 0
        width = 1
        bars = []
        # Plot positive values for all series.
        bottom_values = [0] * num_hours
        for feature_template in feature_templates:
            pos_values = [max(0, weight) for weight in plot_values[feature_template]]
            bar = plt.bar(indices, pos_values,
                    color=colors[series_index % len(colors)],
                    width=width,
                    alpha=0.8,
                    bottom=bottom_values)
            bars.append(bar[0])
            new_bottom_values = [bottom_values[i] + pos_values[i] for i in xrange(num_hours)]
            bottom_values = new_bottom_values
            series_index += 1

        # Plot negative values for all series.
        bottom_values = [0] * num_hours
        for feature_template in feature_templates:
            neg_values = [min(0, weight) for weight in plot_values[feature_template]]
            plt.bar(indices, neg_values,
                    color=colors[series_index % len(colors)],
                    width=width,
                    alpha=0.8,
                    bottom=[i for i in bottom_values])
            new_bottom_values = [bottom_values[i] + neg_values[i] for i in xrange(num_hours)]
            bottom_values = new_bottom_values
            series_index += 1

        # Decorate plot.
        plt.grid(True)
        plt.title('Predicted Number of Pickups in Zone %d' % zone_id)
        plt.xlabel('Time')
        plt.ylabel('Number of Pickups')
        plt.xlim(0, num_hours)
        plt.ylim(-1000, 2000)
        plt.grid(True)
        plt.legend(bars, feature_templates)
        plt.savefig('../outfiles/feature_weights_zone_%d_%s.png' % (zone_id, util.currentTimeString()), bbox_inches='tight')
        plt.close()

    def _plotPredictionError(self, true_num_pickups, predicted_num_pickups):
        '''
        Generates a scatter plot. The true number of pickups is plotted against the prediction error for
        each data point. The prediction error is defined as the absolute difference between the true value
        and the predicted value.
        '''
        util.verbosePrint('Plotting predicted versus true pickup scatter plot.')
        util.verbosePrint('')

        error = [abs(true_num_pickups[i] - predicted_num_pickups[i]) for i in xrange(len(true_num_pickups))]

        # Set area of all bubbles to be 70.
        area = [70]*len(error)
        # plt.scatter(true_num_pickups, error, s=area, alpha=0.2, edgecolors='none')
        plt.scatter(true_num_pickups, predicted_num_pickups, s=area, alpha=0.2, edgecolors='none', label='actual predictions')

        X_line = range(max(true_num_pickups))
        plt.plot(X_line, X_line, 'g--', label='perfect prediction line')
        # Decorate plot.
        plt.grid(True)
        plt.ylabel('Predicted Number of Pickups')
        plt.xlabel('True Number of Pickups')
        plt.title('Predicted vs. True Number of Pickups  \nModel: %s' % self.model)

        plt.legend(loc='best')
        # plt.yscale('log')
        # plt.xscale('log')

        # Hard-code xmin, ymin to be -10, and constrain xmax, ymax to be the greater of the two.
        xmin ,xmax, ymin, ymax = plt.axis()
        plt.axis((-10, max(xmax, ymax), -10, max(xmax, ymax)))
        plt.savefig('../outfiles/true_vs_predicted_scatter_%s.png' % (util.currentTimeString()), bbox_inches='tight')
        plt.close()

    def _plotPredictionHistogram(self, true_num_pickups, predicted_num_pickups):
        '''
        Plots two histograms side-by-side showing the distribution of true and
        predicted number of pickups.
        '''
        util.verbosePrint('Plotting predicted versus true pickup histogram.')
        util.verbosePrint('')

        # Plot histogram.
        num_bins = 30
        plt.hist([predicted_num_pickups, true_num_pickups], num_bins, \
                 label=['Predicted number of pickups', 'True number of pickups'], \
                 alpha=0.5
        )
        plt.yscale('log')
        plt.legend()

        # Label histogram.
        plt.title('Histogram of the number of taxi pickups')
        plt.xlabel('Number of taxi pickups in any zone and hour-long time slot')
        plt.ylabel('Frequency')
        plt.grid(True)

        plt.savefig('../outfiles/true_vs_predicted_histogram_%s.png' % (util.currentTimeString()), bbox_inches='tight')
        plt.close()

def getModel(model_name, database, dataset, options):
    lower_name = model_name.lower()
    if lower_name == 'baseline':
        return Baseline(database, dataset)
    elif lower_name == 'betterbaseline':
        return BetterBaseline(database, dataset)
    elif lower_name == 'linear':
        return LinearRegression(database, dataset)
    elif lower_name == 'svr':
        return SupportVectorRegression(database, dataset)
    elif lower_name == 'dtr':
        return DecisionTreeRegression(database, dataset)
    elif lower_name == 'nnr':
        return NueralNetworkRegression(database, dataset, options.hidden_layer_multiplier)
    elif lower_name == 'autolinear':
        return AutoTunedLinearRegression(database, dataset)
    elif lower_name == 'autodtr':
        return AutoTunedDecisionTree(database, dataset)
    raise Exception('No model with name %s' % model_name)

def getOptions():
    '''
    Get command-line options and handle errors.
    :return: command-line options and arguments
    '''
    parser = OptionParser()
    parser.add_option('-m', '--model', dest='model',
                      help='write report to MODEL', metavar='MODEL')
    parser.add_option('-v', '--verbose',
                      action='store_true', dest='verbose', default=False,
                      help='print verbose output')
    parser.add_option('-n', '--numexamples', type='int', dest='num_examples',
                      default=Const.DATASET_SIZE, help='use a dataset of size NUM',
                      metavar='NUM')
    parser.add_option('-p', '--plot_error',
                      action='store_true', dest='plot_error', default=False,
                      help='generate prediction error scatter plot')
    parser.add_option('-l', '--local',
                      action='store_true', dest='local', default=False,
                      help='use mysql db instance running locally')
    parser.add_option('-f', '--feature_weights_plot',
                      action='store_true', dest='feature_weights', default=False,
                      help='print feature weights for two zones (a high activity and a low activity zone)')
    parser.add_option('--hidden_layer_multiplier', type='float', dest='hidden_layer_multiplier',
                      default=1,
                      help='the size of the neural network hidden layer is INPUT_DIMS * hidden_layer_multipler')
    options, args = parser.parse_args()

    if not options.model:
        print 'Usage: \tpython taxi_pickups.py -m <model-name>'
        print '\nTo see more options, run python taxi_pickups.py --help'
        exit(1)

    if options.verbose:
        util.VERBOSE = True

    return options, args

def main():
    options, args = getOptions()

    database = Database(options.local)
    dataset = Dataset(0.7, options.num_examples, database, Const.AGGREGATED_PICKUPS)
    util.verbosePrint(dataset)

    # Instantiate the specified learning model.
    model = getModel(options.model, database, dataset, options)
    evaluator = Evaluator(model, dataset, options.plot_error, options.feature_weights)

    # Train the model.
    util.verbosePrint('\nTRAINING', model, '...')
    model.train()

    # Evaluate the model on data from the test set.
    util.verbosePrint('\nEVALUATING', model, '...')
    evaluator.evaluate()

if __name__ == '__main__':
    main()
