Description
==============
Repository for our CS221 project for Autumn 2014.

Quick description
=================
If you wish to run our code right away, you can execute the following
to run the linear model on 1000 data points:

```bash
cd src
python taxi_pickups.py -m linear -n 1000 --features feature-sets/features1.cfg -v

# If you get a dependencies error, try running the following:
sudo pip install -r requirements.txt

# Or the following if you are on Stanford's corn machines:
pip install --user MySQL-python==1.2.5
pip install --user pybrain
```

#### MySQL setup
```MySQL
# Note: this setup is only necessary if you want to host the data
# from your local MySQL server as opposed to setting up a remote
# server.

mysql -u root
CREATE DATABASE taxi_pickups;
use taxi_pickups;

# If the data are downloaded as raw pickups, run:
source /path/to/the/sql/script/load-trip-data.sql;
source /path/to/the/sql/script/pickups-aggregated.sql;

# If the data have already been transformed into csv format, run:
source /path/to/the/sql/script/load_pickups_aggregated_from_csv.sql;

# Make sure to modify the sql scripts before running them in order
# to have the correct file paths for your computer.
```

#### Python packages setup
`sudo pip install -r requirements.txt`

#### Running the program
Sample usage: `python taxi_pickups.py -m linear -l -n 1000 --features feature-sets/features1.cfg`  
Run `python taxi_pickups.py --help` for more options.

To submit a job to barley, run:
`python submit_job_barley.py <model parameters>`  
e.g. `python submit_job_barley.py -m autolinear -l -n 1000 -v --features feature-sets/features1.cfg`

#### Connecting to AWS instance to host MySQL database

To copy local MySQL database table to AWS instance:
```bash
sudo mysqldump -u root --single-transaction --compress --order-by-primary taxi_pickups \
pickups_aggregated | mysql -h <instance name>.rds.amazonaws.com -P 3306 -u nyc -p taxi_pickups
```

To connect to the remote MySQL instance from your machine:
```bash
mysql -h <instance name>.rds.amazonaws.com -P 3306 -u nyc -p taxi_pickups
```
