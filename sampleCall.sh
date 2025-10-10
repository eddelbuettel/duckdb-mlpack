#!/bin/bash

# cat <<EOF | build/release/duckdb
# WITH
#   data_query AS (SELECT * from read_csv('data/iris.csv') AS data),
#   lbls_query AS (SELECT * from read_csv('data/iris_labels.txt') AS lbls)
# SELECT *
# FROM read_parquet(
#     (SELECT data FROM data_query),
#     (SELECT lbls FROM lbls_query)
# );
# EOF

test -d data || (echo "Need directory data/ linked to example iris data"; exit 0)

cat <<EOF | build/release/duckdb
CREATE TEMP TABLE Xd AS SELECT * FROM read_csv("data/iris.csv");
CREATE TEMP TABLE X AS SELECT row_number() OVER () AS id, * FROM Xd;
CREATE TEMP TABLE Yd AS SELECT * FROM read_csv("data/iris_labels.txt");
CREATE TEMP TABLE Y AS SELECT row_number() OVER () AS id, CAST(column0 AS double) as label FROM Yd;
CREATE TEMP TABLE D AS SELECT * FROM X INNER JOIN Y ON X.id = Y.id;
ALTER TABLE D DROP id;
ALTER TABLE D DROP id_1;
#SELECT * FROM D;

#CALL mlpack_adaboost((SELECT * FROM D))

CREATE TEMP TABLE A AS SELECT * FROM mlpack_adaboost((SELECT * FROM D));
SELECT COUNT(*) as n, predicted FROM A GROUP BY predicted;
EOF
