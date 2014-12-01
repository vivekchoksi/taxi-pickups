#!/usr/bin/python
import operator
import sys
from feature_extractor import getFeatureNameIndices

VERBOSE = False

def verbosePrint(*args):
    if VERBOSE:
        for arg in args:
           print arg,
        print

def printMostPredictiveFeatures(sklearn_model, n):
    """
    If the input model has feature coefficients, prints the n features whose
    coefficients are the highest, and the n features whose coefficients are
    the lowest.

    :param linear_model: any sklearn_model that has the attribute coef_
    :param n: number of the best/worst features to print (prints 2n features total)
    """
    if not hasattr(sklearn_model, 'coef_'):
        print '\tCannot print out the most predictive features for the model.'
        return

    feature_weights = []
    for feature_name, index in getFeatureNameIndices().iteritems():
        feature_weights.append((feature_name, sklearn_model.coef_[index]))
    feature_weights.sort(key=operator.itemgetter(1))

    def printFeatureWeight(feature_weight):
        print '\t%s:\t%f' % (feature_weight[0], feature_weight[1])

    print ('\tFeature\t\tWeight')
    [printFeatureWeight(feature_weight) for feature_weight in feature_weights[:n]]
    [printFeatureWeight(feature_weight) for feature_weight in feature_weights[-n:]]

def zoneIdToLat(zone_id):
    return (int(zone_id) / 200 + 40 * 100) / 100.0

def zoneIdToLong(zone_id):
    return (int(zone_id) % 200 - 75 * 100) / 100.0

if __name__ == '__main__':
    '''
    Usage: util.py zone_id
    Hacky code to convert zone_id to lat/long coordinates.

    Paste results here to view the region in Google Maps:
    http://www.darrinward.com/lat-long/
    '''
    zone_id = int(sys.argv[1])
    lat = zoneIdToLat(zone_id)
    long = zoneIdToLong(zone_id)
    print 'Zone ID: %d' % int(zone_id)
    print '%.2f,%.2f' % (lat + 0.00, long + 0.00)
    print '%.2f,%.2f' % (lat + 0.01, long + 0.00)
    print '%.2f,%.2f' % (lat + 0.00, long + 0.01)
    print '%.2f,%.2f' % (lat + 0.01, long + 0.01)