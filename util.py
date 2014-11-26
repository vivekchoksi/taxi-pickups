import operator
from feature_extractor import getFeatureNameIndices

VERBOSE = False

def verbosePrint(*args):
    if VERBOSE:
        for arg in args:
           print arg,
        print

def printMostPredictiveFeatures(sklearn_model, n):
    '''
    Prints the n features whose coefficients are the highest, and the n features
    whose coefficients are the lowest.

    :param linear_model: any sklearn_model that has the attributes coef_
    :param n: number of the best/worst features to print (prints 2n features total)
    '''
    feature_weights = []
    for feature_name, index in getFeatureNameIndices().iteritems():
        feature_weights.append((feature_name, sklearn_model.coef_[index]))
    feature_weights.sort(key=operator.itemgetter(1))

    def printFeatureWeight(feature_weight):
        print '%s:\t%f' % (feature_weight[0], feature_weight[1])

    print ('Feature\t\tWeight')
    [printFeatureWeight(feature_weight) for feature_weight in feature_weights[:n]]
    [printFeatureWeight(feature_weight) for feature_weight in feature_weights[-n:]]

def zoneIdToLat(zone_id):
    return (int(zone_id) / 200 + 40 * 100) / 100.0

def zoneIdToLong(zone_id):
    return (int(zone_id) % 200 - 75 * 100) / 100.0
