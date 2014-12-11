#!/usr/bin/python

'''
This file automates submitting a taxi_pickups.py run to barley.

You can change the parameters to the qsub command below.

Sample usage:
    python submit_job_barley.py -m autolinear -n 1000 -v

'''

import os
import sys

args = ' '.join(sys.argv[1:])

# qsub command: submits jobs to the barley queue. For more information, see:
# https://web.stanford.edu/group/farmshare/cgi-bin/wiki/index.php/FarmShare_tutorial
#	qsub parameters:
#   -N filename
#   -l ... what follows is resource request parameters
#   mem_free=1G ... request 1 GB per core
#   -pe shm 4 ... request 4 cores
# os.system('qsub -l mem_free=2G -pe shm 4 -N taxi_pickups_linear run_taxi_pickups.sh')
os.system("qsub -v args='%s' -l mem_free=2G -pe shm 4 -N taxi_pickups_linear " \
    "run_taxi_pickups_barley.sh" % args)

