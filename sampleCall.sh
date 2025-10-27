#!/bin/bash

set -eu
set -o pipefail

## Note that these example show how to access remote data easily with
## duckdb. This requires the (default) extension 'httpfs' and one easy
## way to use it is to set the sub-repos used here to the same version
## as any duckdb binary you may be running.
##
## Otherwise the data are also in this repo in directory docs/data so a
## call such as
##  sed -i -e 's|https://eddelbuettel.github.io/duckdb-mlpack|docs|' sampleCall.sh
## should help. Adjust 'docs' as needed to your local path.

function adaboost {
    cat <<EOF | build/release/duckdb
-- this is needed if you test the locally built extension
SET autoinstall_known_extensions=1;
SET autoload_known_extensions=1;
-- create tables of 'features' and 'labels' followed by override parameters and a model table
CREATE TABLE X AS SELECT * FROM read_csv("https://eddelbuettel.github.io/duckdb-mlpack/data/iris.csv");
CREATE TABLE Y AS SELECT * FROM read_csv("https://eddelbuettel.github.io/duckdb-mlpack/data/iris_labels.csv");
CREATE TABLE Z (name VARCHAR, value VARCHAR);
INSERT INTO Z VALUES ('iterations', '50'), ('tolerance', '1e-7'), ('silent', 'true');
CREATE TABLE M (key VARCHAR, json VARCHAR);

-- train model off 'X' to predict 'Y' using (non-default) parameters in 'Z'
-- serialize model (in JSON) to table 'M'
CREATE TEMP TABLE A AS SELECT * FROM mlpack_adaboost_train("X", "Y", "Z", "M");

-- classification summary of count per group
SELECT COUNT(*) as n, predicted FROM A GROUP BY predicted;

-- Model 'M' can be used to predict on new data so creating 'N'
CREATE TABLE N (x1 DOUBLE, x2 DOUBLE, x3 DOUBLE, x4 DOUBLE);
-- inserting approximate column mean values, min values, max values
INSERT INTO N VALUES (5.843, 3.054, 3.759, 1.199), (4.3, 2.0, 1.0, 0.1), (7.9, 4.4, 6.9, 2.5);
-- and this predict one element each
SELECT * FROM mlpack_adaboost_pred("N", "M");

EOF
}

function linear_regression {
    cat <<EOF | build/release/duckdb
SET autoinstall_known_extensions=1;
SET autoload_known_extensions=1;
CREATE TABLE X AS SELECT * FROM read_csv("https://eddelbuettel.github.io/duckdb-mlpack/data/trees_x.csv");
CREATE TABLE Y AS SELECT * FROM read_csv("https://eddelbuettel.github.io/duckdb-mlpack/data/trees_y.csv");
CREATE TABLE Z (name VARCHAR, value VARCHAR);
INSERT INTO Z VALUES ('intercept', 'true');
CREATE TABLE M (key VARCHAR, json VARCHAR);

SELECT * FROM mlpack_linear_regression_fit("X", "Y", "Z", "M");

SELECT * FROM M WHERE key = 'coefficients';

## Create a quick new data table with arbitrary values and compute prediction on new data
## This replicates what we see in R with these values
CREATE TABLE N (x1 DOUBLE, x2 DOUBLE);
INSERT INTO N VALUES ('1', '1'), ('2', '-1'), ('3', '1');
SELECT * FROM mlpack_linear_regression_pred("N", "M");

DROP TABLE X;
DROP TABLE Y;
DROP TABLE Z;
DROP TABLE M;

EOF
}

function linear_regression_larger {
    cat <<EOF | build/release/duckdb
SET autoinstall_known_extensions=1;
SET autoload_known_extensions=1;
CREATE TABLE X AS SELECT * FROM read_csv("https://eddelbuettel.github.io/duckdb-mlpack/data/covertype_small_features.csv.gz");
CREATE TABLE Y AS SELECT * FROM read_csv("https://eddelbuettel.github.io/duckdb-mlpack/data//covertype_small_labels.csv.gz");
CREATE TABLE Z (name VARCHAR, value VARCHAR);
INSERT INTO Z VALUES ('intercept', 'false'), ('lambda', '0.05');
CREATE TABLE M (key VARCHAR, json VARCHAR);

CREATE TEMP TABLE A AS SELECT * FROM mlpack_linear_regression_fit("X", "Y", "Z", "M");
SELECT * FROM M WHERE key = 'coefficients';

EOF
}

adaboost
linear_regression
linear_regression_larger
