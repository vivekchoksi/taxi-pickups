Description
==============
Repository for our CS221 project for Autumn 2014.

MySQL setup:
==============
mysql -u root
CREATE DATABASE taxi_pickups;
use taxi_pickups;
source /path/to/the/sql/script/load-trip-data.sql

Python packages setup
======================
sudo pip install -r requirements.txt

Running the program
====================
Sample usage: python taxi_pickups.py -m linear -n 1000 --features features1.cfg
Run python taxi_pickups.py --help for more options

To submit a job to barley, run:
python submit_job_barley.py [model params-- e.g. -m autolinear -n 1000 -v --features features1.cfg]

Useful Commands
=================

Password for connecting to MySQL db: gottapickthemall

To copy local MySQL database table to AWS instance:
sudo mysqldump -u root --single-transaction --compress --order-by-primary taxi_pickups pickups_aggregated | mysql -h taxi-pickups.cw6ohvqgsy0r.us-west-1.rds.amazonaws.com -P 3306 -u nyc -p taxi_pickups

To connect to the remote MySQL instance from your machine:
mysql -h taxi-pickups.cw6ohvqgsy0r.us-west-1.rds.amazonaws.com -P 3306 -u nyc -p taxi_pickups

