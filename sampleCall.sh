#!/bin/bash

cat <<EOF | build/release/duckdb
SET autoinstall_known_extensions=1;
SET autoload_known_extensions=1;
CREATE TABLE X AS SELECT * FROM read_csv("https://mlpack.org/datasets/iris.csv");
CREATE TABLE Y AS SELECT * FROM read_csv("https://mlpack.org/datasets/iris_labels.csv");
CREATE TABLE Z (name VARCHAR, value VARCHAR);
INSERT INTO Z VALUES ('iterations', '5'), ('tolerance', '2e-6'), ('verbose', 'true');

CREATE TEMP TABLE A AS SELECT * FROM mlpack_adaboost("X", "Y", "Z");

SELECT COUNT(*) as n, predicted FROM A GROUP BY predicted;

DROP TABLE X;
DROP TABLE Y;
DROP TABLE Z;
EOF
