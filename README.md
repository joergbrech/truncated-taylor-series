# Truncated Taylor series Approximation

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/joergbrech/truncated-taylor-series/HEAD?urlpath=voila/render/taylor_approx.ipynb)
[![License: CC BY-SA 4.0](https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-sa/4.0/)

This is a simple web app demonstrating the truncated Taylor series approximation of a function. [Click here](https://mybinder.org/v2/gh/joergbrech/truncated-taylor-series/HEAD?urlpath=voila/render/taylor_approx.ipynb) to launch the app. This app is built using a jupyter notebook and voila.

![](screenshot.PNG)

To run locally, it is recommended to install all requirements in a conda environment. To do so, install an Anaconda distribution if you haven't already, e.g. [Miniforge](https://github.com/conda-forge/miniforge), and open a conda terminal. Enter the following commands to create the environment, install all requirements and activate the environment:

```
conda env create -f .binder/environment.yml
conda activate rise-matplotlib
```

Next, copy the commands from `.binder/postBuild` into the same terminal to setup all necessary jupyter notebook extensions. If you are running a bash-like terminal, you can simply run the following command to do this.

```
source .binder/postBuild
```

Finally, start the app with

```
voila taylor_approx.ipynb
```
