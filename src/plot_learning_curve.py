#!/usr/bin/python
"""
========================
Plotting Learning Curves
========================
"""
import os
os.environ['MPLCONFIGDIR'] = "../"
import numpy as np
import matplotlib.pyplot as plt
import util
from optparse import OptionParser
from sklearn.learning_curve import learning_curve
from taxi_pickups import *


def plotLearningCurve(model, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and traning learning curve.

    Parameters
    ----------
    model : RegressionModel that contains a regressor object as a field.
        This regressor object is cloned for each validation. It must support 
        "fit" and "predict" methods.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : integer, cross-validation generator, optional
        If an integer is passed, it is the number of folds (defaults to 3).
        Specific cross-validation objects can be passed, see
        sklearn.cross_validation module for the list of possible objects

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        model.regressor, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    print train_scores
    print test_scores
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Testing score")

    plt.legend(loc="best")
    return plt


def getModel(model_name, database, dataset):
    lower_name = model_name.lower()
    if lower_name == 'linear':
        return LinearRegression(database, dataset)
    elif lower_name == 'svr':
        return SupportVectorRegression(database, dataset)
    elif lower_name == 'dtr':
        return DecisionTreeRegression(database, dataset)
    raise Exception('Model with name %s not supported for learning curves.' % \
        model_name)

def extractDataset(dataset, model_name):
    feature_extractor = FeatureExtractor(model_name != 'dtr')
    row_dicts = []
    while dataset.hasMoreTrainExamples():
        row_dicts.extend(dataset.getTrainExamples(Const.TRAIN_BATCH_SIZE))

    dataset.switchToTestMode()
    while dataset.hasMoreTestExamples():
        row_dicts.append(dataset.getTestExample())

    util.verbosePrint('Number of examples being considered for train and test:',
        len(row_dicts))

    X = feature_extractor.getFeatureVectors(row_dicts)
    y = np.array([example['num_pickups'] for example in row_dicts])
    return (X, y)

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
    parser.add_option('-i', '--iterations', type='int',
                      dest='num_iter', default=5,
                      help='number of runs for each data set size to average over')
    parser.add_option('-f', '--train_fraction', type='float',
                      dest='train_fraction', default=0.7,
                      help='fraction of num_examples to use as training data')
    options, args = parser.parse_args()

    if not options.model:
        print 'Usage: \tpython plot_learning_curve.py -m <model-name>'
        print '\nTo see more options, run python plot_learning_curve.py --help'
        exit(1)

    if options.verbose:
        util.VERBOSE = True

    return options, args

def main():
    options, args = getOptions()
    database = Database()
    dataset = Dataset(0.7, options.num_examples, database, 
        Const.AGGREGATED_PICKUPS)
    X, y = extractDataset(dataset, options.model)
    model = getModel(options.model, database, dataset)
    title = "Learning Curves (%s)" % model
    cv = util.getCrossValidator(options.num_iter, options.train_fraction, 
        options.num_examples)
    plotLearningCurve(model, title, X, y, ylim=(0.5, 1.01), cv=cv, n_jobs=4)
    plt.show()


if __name__ == '__main__':
    main()
