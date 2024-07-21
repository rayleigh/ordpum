This repository contains code needed to run the ordinal probit unfolding model. In order to do so, the package `ordpum` needs to be installed first in R. This can be accomplished using the following commands:

```
library(devtools)
library(Rcpp)

Rcpp::compileAttributes("code/ordpum/")
devtools::document("code/ordpum/")
install("code/ordpum/")
```

Once `ordpum` has been installed, there are R scripts to analyze an immigration battery from Duck-Mayr and Montgomery's 2022 paper, Ends Against the Middle: Measuring Latent Traits When Opposites Respond the Same Way for Antithetical Reasons, and a COVID attitude and trust survey from Ridenhour et al's 2022 paper, Effects of trust, risk perception, and health behavior on COVID-19 disease burden: Evidence from a multi-state US survey. Both data sets are included as well in `data_files`. As a comparison, the Stan code for the graded response model and the data sets in .json format are included as well.
