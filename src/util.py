#!/usr/bin/python
import sys, random
import numpy as np

VERBOSE = False

def verbosePrint(*args):
    if VERBOSE:
        for arg in args:
           print arg,
        print

def zoneIdToLat(zone_id):
    return (int(zone_id) / 200 + 40 * 100) / 100.0

def zoneIdToLong(zone_id):
    return (int(zone_id) % 200 - 75 * 100) / 100.0

def getCrossValidator(num_iter, train_fraction, num_examples):
    verbosePrint(
        'Num Iterations: %d\n' % num_iter, 
        'Train Fraction: %0.2f\n' % train_fraction, 
        'Num Examples passed in: %d\n' % num_examples
    )
    max_id = num_examples
    max_train_id = int(train_fraction * max_id)
    test_ids = np.array(range(max_train_id, max_id))
    cv = []

    for _ in range(num_iter):
        train_ids = range(max_train_id)
        random.shuffle(train_ids)
        cv.append((np.array(train_ids), test_ids))

    # for train_indices, test_indices in cv:
    #     verbosePrint("Train:", train_indices)
    #     verbosePrint("Test:", test_indices)

    return cv

if __name__ == '__main__':
    '''
    Usage: util.py zone_id
    Hacky code to convert zone_id to lat/long coordinates.

    Paste results here to view the box region in Google Maps:
    http://www.darrinward.com/lat-long
    '''
    zone_id = int(sys.argv[1])
    lat = zoneIdToLat(zone_id)
    long = zoneIdToLong(zone_id)
    print 'Zone ID: %d' % int(zone_id)
    print '%.2f,%.2f' % (lat + 0.00, long + 0.00)
    print '%.2f,%.2f' % (lat + 0.01, long + 0.00)
    print '%.2f,%.2f' % (lat + 0.00, long + 0.01)
    print '%.2f,%.2f' % (lat + 0.01, long + 0.01)