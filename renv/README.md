# R reproducibility

R is supported when installed. Initialize a project-local lockfile with:

```r
install.packages('renv')
renv::init(bare=TRUE)
renv::install(c('tidyverse','broom','car','knitr','rmarkdown'))
renv::snapshot()
```

Then use `renv::restore()` on another machine and run `Rscript R/statistical_analysis.R`. An `renv.lock` file is not fabricated when R is unavailable.
