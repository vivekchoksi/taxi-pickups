#!/usr/bin/python

import os
os.environ['MPLCONFIGDIR'] = '../'
import random
import datetime
from math import sqrt
import sklearn.metrics as metrics
import numpy as np
import util
from const import Const
import matplotlib

# Quirky command to execute before importing matplotlib.pyplot; necessary for
# running plotting code on Stanford Barley machines.
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# The Evaluator class evaluates a trained model.
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

        # Plot the feature weights specific to a zone.
        if self.print_feature_weights:
            start_datetime = datetime.datetime(2013, 4, 7)
            self._plotFeatureWeights(14901, start_datetime)
            self._plotFeatureWeights(16307, start_datetime)

        # Evaluate the predictions.
        self._evaluatePredictions(true_num_pickups, predicted_num_pickups)

        # Plot the prediction error versus true number of pickups.
        if self.plot_error:
            self._plotPredictionError(true_num_pickups, predicted_num_pickups)

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

        # Compute and print the coefficient of determination, R^2.
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
        #   e.g. feature_weights['Zone_HourOfDay=15402_14'] = 324.4565
        feature_weights = self.model.getFeatureWeights()
        if feature_weights is None:
            print '\tAborting feature weight plot.'
            return

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

        colors = ['b', 'g', 'r', 'c', 'y', 'm', '0.2', '0.8']
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

        series_index = 0
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
        plt.xlabel('Time (hours since 2013 April 7, 12am)')
        plt.ylabel('Number of Pickups')
        plt.xlim(0, num_hours)
        plt.ylim(-1000, 2000)
        plt.xticks(np.arange(0, num_hours + 1, 12))
        plt.grid(True)
        plt.legend(bars, feature_templates)
        plt.savefig('../outfiles/feature_weights_zone_%d_%s.png' % (zone_id, util.currentTimeString()), bbox_inches='tight')
        # plt.show()
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
        plt.plot(X_line, X_line, 'g--', color='0.5', label='perfect prediction line')
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
