#!/bin/bash

# Tell grid engine to merge stdout and stderr streams.
#$ -j y

# Mail to this address.
#$ -M vhchoksi@stanford.edu
# Send mail on beginning, ending, or suspension of job.
#$ -m bes

# Tell grid engine what directory to use.
# cwd means current directory.
#$ -cwd

echo ${args}

# Install MySQLdb dependency.
pip install --user MySQL-python==1.2.5

pip install --user pybrain

# CONFIGURE THIS TO BE THE COMMAND YOU WANT TO RUN.
time python taxi_pickups.py ${args}

