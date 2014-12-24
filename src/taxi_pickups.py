#!/usr/bin/python

'''
Train and evaluate a regression model.

Sample usage:
    python taxi_pickups.py -m linear -l -v --features feature-sets/features1.cfg

Run python taxi_pickups.py --help for more information.
'''

import os
from optparse import OptionParser
from models import *
from data_management import *
from evaluator import Evaluator


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
        return NeuralNetworkRegression(database, dataset, options.hidden_layer_multiplier)
    elif lower_name == 'autolinear':
        return AutoTunedLinearRegression(database, dataset)
    elif lower_name == 'autodtr':
        return AutoTunedDecisionTree(database, dataset)
    elif lower_name == 'autosvr':
        return AutoTunedSVR(database, dataset)
    raise Exception('No model with name %s' % model_name)


def getOptions():
    '''
    Get command-line options and handle errors.
    :return: command-line options and arguments
    '''
    parser = OptionParser()
    parser.add_option('-m', '--model', dest='model',
                      help='regression model to use: baseline, betterbaseline, ' +
                           'linear, svr, dtr, nnr, autolinear, autodtr, or autosvr')
    parser.add_option('--features', dest='features_file',
                      help='name of the features config file; e.g. feature-sets/features1.cfg')
    parser.add_option('-v', '--verbose',
                      action='store_true', dest='verbose', default=False,
                      help='print verbose output')
    parser.add_option('-n', '--numexamples', type='int', dest='num_examples',
                      default=sys.maxint, help='use a dataset limited to size NUM',
                      metavar='NUM')
    parser.add_option('--maxtrainexamples', type='int', dest='max_train_examples',
                      default=sys.maxint, help='maximum number of examples on which to train')
    parser.add_option('-p', '--plot_error',
                      action='store_true', dest='plot_error', default=False,
                      help='generate prediction error scatter plot')
    parser.add_option('-l', '--local',
                      action='store_true', dest='local', default=False,
                      help='use local as opposed to remote MySQL server')
    parser.add_option('-f', '--feature_weights_plot',
                      action='store_true', dest='feature_weights', default=False,
                      help='print feature weights for two zones (a high activity and a low activity zone)')
    parser.add_option('--hidden_layer_multiplier', type='float', dest='hidden_layer_multiplier',
                      default=1,
                      help='the size of the neural network hidden layer is INPUT_DIMS * hidden_layer_multipler')
    options, args = parser.parse_args()

    if not options.model or not options.features_file:
        print 'Usage: \tpython taxi_pickups.py -m <model-name> --features <features-filename.cfg>'
        print '\nTo see more options, run python taxi_pickups.py --help'
        exit(1)

    if os.path.splitext(options.features_file)[1] != '.cfg' or \
            not os.path.isfile(options.features_file):
        print 'Invalid feature file name. Must be a path to a valid .cfg file.'
        print '\nTo see more options, run python taxi_pickups.py --help'
        exit(1)

    if options.verbose:
        util.VERBOSE = True

    return options, args


def main():
    options, args = getOptions()

    database = Database(options.local)
    dataset = Dataset(0.7, options.num_examples, options.max_train_examples,
                      database, Const.AGGREGATED_PICKUPS)
    util.verbosePrint(dataset)

    util.FEATURES_FILE = options.features_file

    # Instantiate the specified learning model.
    model = getModel(options.model, database, dataset, options)
    evaluator = Evaluator(model, dataset, options.plot_error,
                          options.feature_weights)

    # Train the model.
    util.verbosePrint('\nTRAINING', model, '...')
    model.train()

    # Evaluate the model on data from the test set.
    util.verbosePrint('\nEVALUATING', model, '...')
    evaluator.evaluate()

if __name__ == '__main__':
    main()
