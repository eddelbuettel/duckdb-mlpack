
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
first creates a temporary tables for the two data sets for 'features' and 'labels' (or `X` and `y`
in more statistical terminology) which are passed a strings to the function and read therein.  It 
also uses a parameter table `Z` with simply `key: value` notation as variable charater entries. Keys
corresponding to parameters will override defaults with their given values. Lastly, table `M`
contains the (JSON-)serialized model one could use for prediction on new data. 

```sh
#!/bin/bash

cat <<EOF | build/release/duckdb
SET autoinstall_known_extensions=1;
SET autoload_known_extensions=1; # for httpfs

CREATE TABLE X AS SELECT * FROM read_csv("https://mlpack.org/datasets/iris.csv");
CREATE TABLE Y AS SELECT * FROM read_csv("https://mlpack.org/datasets/iris_labels.csv");
CREATE TABLE Z (name VARCHAR, value VARCHAR);
INSERT INTO Z VALUES ('iterations', '150'), ('tolerance', '1e-8'), ('verbose', 'true');
CREATE TABLE M (json VARCHAR);

CREATE TEMP TABLE A AS SELECT * FROM mlpack_adaboost("X", "Y", "Z", "M");

SELECT COUNT(*) as n, predicted FROM A GROUP BY predicted;

DROP TABLE X;
DROP TABLE Y;
DROP TABLE Z;
DROP TABLE M;
EOF
```

Note also how we take advantage of the remote `httpfs` reader in `duckdb`; we could equally well
read a local file, or one from S3, or ... We ensured the two git submodules used where at the same
version as our normal `duckdb` executable, currently 1.4.1, so that we could take advange of the
`httpfs` extension used to do the remote read. If you not have the extension you can also use,
respectively, `read_csv('path/to/iris.csv')` and `read_csv('path/to/iris_labels.csv')`.)

Running this script (after building the extension) results in 

```sh
$ ./sampleCall.sh 
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
`duckdb` extensions for more. `mlpack` and its dependencies will also be as needed during the build
installed,


## TODO

- [Partionally DONE: linear regression] More examples of model fitting and prediction
- [Partionally DONE] Maybe set up model serialization into table to predict on new data
- Ideally: Work out how to `SELECT` from multiple tabels
- [DONE] Else maybe `SELECT` into temp. tables and pass temp. table names into routine
- [DONE] Read parameters from auxiliary table
- ~~Maybe add `mlpack` as a `git submodule`~~ CMake now pulls it in as a dependency 

## Acknowledgements

Ryan Curtin has been very helpful during design discussions and with endless CMake tips.

## Author

Dirk Eddelbuettel for this repo

The duckdb authors for `duckdb`

The mlpack authors for `mlpack`

## License

MIT 

