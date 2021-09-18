# Truncated Taylor series Approximation

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/joergbrech/truncated-taylor-series/HEAD?urlpath=/tree/taylor_approx.ipynb)
[![License: CC BY-SA 4.0](https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-sa/4.0/)

This is a jupyter notebook with a simple widget demonstrating the truncated Taylor series approximation of a function. [Click here](https://mybinder.org/v2/gh/joergbrech/truncated-taylor-series/HEAD?urlpath=/tree/taylor_approx.ipynb) to launch the jupyter notebook interactively.

![](screenshot.PNG)

To run locally, it is recommended to install all requirements in a conda environment. Install an Anaconda distribution, e.g. [Miniforge](https://github.com/conda-forge/miniforge) and open an Anaconda terminal. 

```
conda env create -f .binder/environment.yml
conda activate rise-matplotlib
```

Next, copy the commands from `.binder/postBuild` into the same terminal to setup all necessary jupyter notebook extensions. If you are running a bash-like terminal, you can simply run 

```
source .binder/postBuild
```

Finally, start the jupyter notebook with

```
jupyter taylor_approx.ipynb
```
