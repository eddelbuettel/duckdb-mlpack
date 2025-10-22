#!/bin/bash

function adaboost {
    cat <<EOF | build/release/duckdb
SET autoinstall_known_extensions=1;
SET autoload_known_extensions=1;
CREATE TABLE X AS SELECT * FROM read_csv("https://mlpack.org/datasets/iris.csv");
CREATE TABLE Y AS SELECT * FROM read_csv("https://mlpack.org/datasets/iris_labels.csv");
CREATE TABLE Z (name VARCHAR, value VARCHAR);
INSERT INTO Z VALUES ('iterations', '5'), ('tolerance', '2e-6'), ('verbose', 'true');
CREATE TABLE M (json VARCHAR);

CREATE TEMP TABLE A AS SELECT * FROM mlpack_adaboost("X", "Y", "Z", "M");

SELECT COUNT(*) as n, predicted FROM A GROUP BY predicted;
EOF
}

function linear_regression {
    ## this needs local data files for now; these correspond the (log) values
    ## of the columns in R's trees data set, see
    ## https://github.com/eddelbuettel/rcppmlpack-examples/blob/master/man/linearRegression.Rd#L32-L34
    cat <<EOF | build/release/duckdb
CREATE TABLE X AS SELECT * FROM read_csv("local/trees_x.csv");
CREATE TABLE Y AS SELECT * FROM read_csv("local/trees_y.csv");
CREATE TABLE Z (name VARCHAR, value VARCHAR);
INSERT INTO Z VALUES ('intercept', 'true');
CREATE TABLE M (json VARCHAR);

CREATE TEMP TABLE A AS SELECT * FROM mlpack_linear_regression_fit("X", "Y", "Z", "M");

SELECT * FROM A;
SELECT * FROM M;

CREATE TEMP TABLE B AS SELECT * FROM mlpack_linear_regression_pred("X", "M");
SELECT * FROM B;
EOF
}

adaboost
#linear_regression
