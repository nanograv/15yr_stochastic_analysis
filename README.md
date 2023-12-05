# NANOGrav_15yr_tutorial <img align="right" width="150" height="100" src="https://github.com/nanograv/12p5yr_stochastic_analysis/blob/master/nanograv.png?raw=true">
## Tutorial to go with the [15 year isotropic GWB analysis](https://arxiv.org/abs/2306.16213)

[![Generic badge](https://img.shields.io/badge/Created%20by-NANOGrav-red)](http://nanograv.org/)
[![MIT license](https://img.shields.io/badge/License-MIT-blue.svg)](https://lbesson.mit-license.org/)
[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
[![Generic badge](https://img.shields.io/twitter/follow/NANOGrav?style=social)](https://twitter.com/NANOGrav)
 
Authors: [Aaron Johnson](https://github.com/AaronDJohnson), [Stephen Taylor](http://stevertaylor.github.io/), [Sarah Vigeland](https://github.com/svigeland) for the [NANOGrav Collaboration](https://github.com/nanograv)

Please send questions about this tutorial to `aaron.johnson (at) nanograv.org`

## <span style="color:red">Important Note About Data!</span>
**If you want to use our data for publications, the full data are available for download here (add link here).**

Data used in the `tutorials` section of this repository have been reduced to make them available on GitHub and may not reproduce the results of the 15-year analysis exactly.

## Installing PTA software

1. Install Miniconda

    * Download and install the relevant Miniforge file from here: https://github.com/conda-forge/miniforge


2. To install a new environment: `conda create -n enterprise -c conda-forge enterprise_extensions la_forge h5pulsar ipympl jupyterlab`
    * Newer Macs (M1/M2/M3) can make a conda environment and install `enterprise` by first following the instructions [here](https://conda-forge.org/docs/user/tipsandtricks.html#installing-apple-intel-packages-on-apple-silicon) to create an environment and then using `conda install -c conda-forge enterprise_extensions la_forge h5pulsar ipympl jupyterlab`

3. This will create a conda environment that can be activated by `conda activate enterprise`

4. Next run `jupyter notebook`

5. Set the Kernel

    * when opening a new notebook: click `New` and select `Python [conda env:enterprise]`  
    * when opening an existing notebook (like these tutorials): click `Kernel` --> `Change Kernel` --> `Python [conda env:enterprise]`  

## Tutorials

  These tutorials are split into several different files. The topic of each tutorial is shown below. These are roughly in the order that they should be viewed in to get a complete picture of how the isotropic GWB analysis is performed in NANOGrav.

### Exploring NANOGrav Data

  NANOGrav uses data files that may be unfamiliar to users that are new to pulsar timing or data analysis. Here, we investigate what information exists inside each `par` and `tim` file, how to load them into `enterprise`, and what information `enterprise` can use.

### PTA GWB Analysis
  
  This tutorial is split into two files, one for parameter estimation and one for model selection. We work through the Bayesian analysis of the full NANOGrav 15 year data to show what values each of the searched-over parameters prefers. Next, we show how to compare models and compute Bayes factors.

### Optimal Statistic Analysis
	
  This tutorial gives an introduction to frequentist methods we can use to look for an isotropic GWB.
