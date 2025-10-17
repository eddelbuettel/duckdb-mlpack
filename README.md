
# duckdb-mlpack: Accessing mlpack from duckdb

## About

`duckdb` is a very versatile database with fairly universal data access abilities.  `mlpack` is an
excellent machine learning library.

The two fit like peanut butter and jelly. `duckdb` is written in C++ and self-contained. All
external dependencies are 'vendored' (i.e. pulled into the source) creating a single C++ executable
(or library to interface from C++ and many other languages). `mlpack` is also written in C++, with
minimal external dependencies (to Armadillo and Ensmallen, which can be use header-only or as small
libraries).  As `mlpack` does not require another (expensive) run-time, it is ideal for 'smaller'
and self-contained used such as embedded system, or efficient binaries. Given the ability to embed,
it is also an ideal candidate for a `duckdb` extension.

This repository provides just that. It uses the excellent `duckdb` extension template and extends it
minimally with an example function to run the Adaboost classifier.

It is currently in 'MVP' status: _minimal_ indeed but also _viable_ as the following example shows.

## Example

Because table functions in `duckdb` can only use on `select *` query, the following code snippet
first creates a temporary table joining the two data sets for 'features' and 'labels' (or `X` and
`y` in more statistical terminology) before reading from the combined table.  Note also how we 
take advantage of the remote `httpfs` reader in `duckdb`; we could equally well read a local file,
or one from S3, or ...

```sh
#!/bin/bash

cat <<EOF | build/release/duckdb
SET autoinstall_known_extensions=1;
SET autoload_known_extensions=1; # for httpfs

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
```

(Note that we ensured the two git submodules used where at the same version as our normal `duckdb`
executable so that we could take advange of the `httpfs` extension used to do the remote read. If
you not have the extension you can also use, respectively, `read_csv('path/to/iris.csv')` and
`read_csv('path/to/iris_labels.csv')`.)

Running this script (after building the extension) results in 

```sh
$ ./sampleCallRemote.sh 
Misclassified: 1
┌───────┬───────────┐
│   n   │ predicted │
│ int64 │   int32   │
├───────┼───────────┤
│    50 │         0 │
│    49 │         1 │
│    51 │         2 │
└───────┴───────────┘
$   
```

which is the result we get from, say, the `adaboost` example running on the same data via the R
extension (from [this repo](https://github.com/eddelbuettel/rcppmlpack-examples)): one misclassified
entry (when predicting the on training data -- the example here is _minimal_):

```r
> X <- t(as.matrix(read.csv("https://mlpack.org/datasets/iris.csv")))
> y <- as.integer(read.csv("https://mlpack.org/datasets/iris_labels.csv", header=FALSE)[,1]) - 1L
> res <- rcppmlpackexamples::adaBoost(X, y)
> table(res$predicted)

 0  1  2 
50 49 51 
>
```

## Installation

Requirements should be the same as for `duckdb` and `mlpack`: a recent compiler, and for the latter
also a local Armadillo installation as we have not told `cmake` yet to install Armadillo if not
found.

Then just say

```sh
make
```

and after a short little while `build/release/duckdb` should be available. See the documentation for
`duckdb` extensions for more.

## TODO

- More examples of model fitting and prediction
- Maybe set up model serialization into table to predict on new data
- Ideally: Work out how to `SELECT` from multiple tabels, or else maybe `SELECT` into temp. tables
  and pass temp. table names into routine
- Maybe add `mlpack` as a `git submodule` 

## Author

Dirk Eddelbuettel for this repo

The duckdb authors for `duckdb`

The mlpack authors for `mlpack`

## License

MIT 

