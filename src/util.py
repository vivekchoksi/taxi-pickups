#!/usr/bin/python
import sys

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