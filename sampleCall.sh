#!/bin/bash

cat <<EOF | build/release/duckdb
SET autoinstall_known_extensions=1;
SET autoload_known_extensions=1;
CREATE TABLE X AS SELECT * FROM read_csv("https://mlpack.org/datasets/iris.csv");
CREATE TABLE Y AS SELECT * FROM read_csv("https://mlpack.org/datasets/iris_labels.csv");

CREATE TEMP TABLE A AS SELECT * FROM mlpack_adaboost("X", "Y");

SELECT COUNT(*) as n, predicted FROM A GROUP BY predicted;

DROP TABLE X;
DROP TABLE Y;
EOF
