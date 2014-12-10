#!/usr/bin/python

import os

# qsub command: submits jobs to the barley queue. For more information, see:
# https://web.stanford.edu/group/farmshare/cgi-bin/wiki/index.php/FarmShare_tutorial
# 	qsub parameters:
#	-N filename
#	-l ... what follows is resource request parameters
#	mem_free=1G ... request 1 GB per core
#	-pe shm 4 ... request 4 cores
# os.system('qsub -l mem_free=2G -pe shm 4 -N taxi_pickups_linear run_taxi_pickups.sh')
os.system('qsub -N taxi_pickups_linear run_taxi_pickups.sh')

