#!/bin/bash

cat <<EOF | build/release/duckdb
SET autoinstall_known_extensions=1;
SET autoload_known_extensions=1;
CREATE TEMP TABLE Xd AS SELECT * FROM read_csv("https://mlpack.org/datasets/iris.csv");
CREATE TEMP TABLE X AS SELECT row_number() OVER () AS id, * FROM Xd;
CREATE TEMP TABLE Yd AS SELECT * FROM read_csv("https://mlpack.org/datasets/iris_labels.csv");
CREATE TEMP TABLE Y AS SELECT row_number() OVER () AS id, CAST(column0 AS double) as label FROM Yd;
CREATE TEMP TABLE D AS SELECT * FROM X INNER JOIN Y ON X.id = Y.id;
ALTER TABLE D DROP id;
ALTER TABLE D DROP id_1;
CREATE TEMP TABLE A AS SELECT * FROM mlpack_adaboost((SELECT * FROM D));

SELECT COUNT(*) as n, predicted FROM A GROUP BY predicted;

EOF
