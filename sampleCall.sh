#!/bin/bash

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
SET autoinstall_known_extensions=1;
SET autoload_known_extensions=1;
CREATE TABLE X AS SELECT * FROM read_csv("https://eddelbuettel.github.io/duckdb-mlpack/data/iris.csv");
CREATE TABLE Y AS SELECT * FROM read_csv("https://eddelbuettel.github.io/duckdb-mlpack/data/iris_labels.csv");
CREATE TABLE Z (name VARCHAR, value VARCHAR);
INSERT INTO Z VALUES ('iterations', '5'), ('tolerance', '2e-6'), ('verbose', 'true');
CREATE TABLE M (json VARCHAR);

CREATE TEMP TABLE A AS SELECT * FROM mlpack_adaboost("X", "Y", "Z", "M");

SELECT COUNT(*) as n, predicted FROM A GROUP BY predicted;
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
CREATE TABLE M (json VARCHAR);

CREATE TEMP TABLE A AS SELECT * FROM mlpack_linear_regression_fit("X", "Y", "Z", "M");

## Checks
SELECT * FROM A;
SELECT * FROM M;

## For simplicity re-fit from given data
SELECT * FROM (SELECT * FROM mlpack_linear_regression_pred("X", "M"));

## Create a quick new data table with arbitrary values and compute prediction on new data
## This replicates what we
CREATE TABLE N (x1 DOUBLE, x2 DOUBLE);
INSERT INTO N VALUES ('1', '1'), ('2', '-1'), ('3', '1');
SELECT * FROM (SELECT * FROM mlpack_linear_regression_pred("N", "M"));

DROP TABLE X;
DROP TABLE Y;
DROP TABLE Z;
DROP TABLE A;
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
CREATE TABLE M (json VARCHAR);

CREATE TEMP TABLE A AS SELECT * FROM mlpack_linear_regression_fit("X", "Y", "Z", "M");
SELECT * FROM M;

EOF
}

#adaboost
#linear_regression
linear_regression_larger
