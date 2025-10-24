
## Overview

Following [mlpack][mlpack] convention we split data sets that may be in one common data frame
containing both predictors and dependent variable into two as this faciliates loading into two
distinct (temporary) tables from which the functions added here read them.

## Data Sets

### Trees

This is a variant of the standard R example where the `trees` data set is used in logs.  We simply
save `X` and `y` after a simple `log()` transformation shown in some other R examples, i.e.

```r
> data(trees)
> X <- with(trees, cbind(log(Girth), log(Height)))
> write.csv(X, "trees_x.csv", row.names=FALSE)
>
y <- with(trees, cbind(log(Volume)))
> write.csv(y, "trees_y.csv", row.names=FALSE)
> 
```

The reference given in `help(trees)` is Atkinson, A. C. (1985) _Plots, Transformations and
Regression_.  Oxford University Press.  The help pages actually a different multiplicative model for
tree volume, we really only this as a minimal example.


### Iris

This well known data set is included in R (see `help(iris)`) and the [UC Irvine data repository for
machine learning][ucimldata] and is described on [this page][iris] in more detail. We copied the
[mlpack data][mldata] files that already split into features and labels.


### Covertype

This is standard [mlpack][mlpack] example for random forests. The data set originates from the [UC
Irvine data repository for machine learning][ucimldata] and is described on [this
page][covertype] in more detail. We took the subset containing 10k rows from the [mlpack data
page][mlpackdata], i.e. we did not sample ourselves. We split the 55th column `labels` off into its
own file, and kept the other 54.  The dimensions are now 10,000 x 54 and 10,000 x 1, respectively.

     
     
[mlpack]: https://mlpack.org
[mlpackdata]: https://mlpack.org/datasets/
[ucimldata]: https://archive.ics.uci.edu/
[covertype]: https://archive.ics.uci.edu/dataset/31/covertype
[iris]: https://archive.ics.uci.edu/dataset/53/iris
