![Static Badge](https://img.shields.io/badge/python-3.9-blue)　![Static Badge](https://img.shields.io/badge/license-MIT-orange)　![Static Badge](https://img.shields.io/badge/DOI-to%20be%20supplemented-purple)　![Static Badge](https://img.shields.io/badge/Zenodo-to%20be%20supplemented-green)
# AL_for_SL_SR
- [Abstract](#abstract) 
- [Mpdules](#modules)
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)

## Abstract
To be supplemented
## Modules
* `model.py`
  Implements the predictive models used in this study, together with the ensemble framework for uncertainty estimation and robust prediction.
* `Handler.py`
  Manages the active learning process by maintaining and updating the labeled training set and unlabeled sample pool throughout each learning cycle.
* `experiment.py`
  Provides high-level interfaces that integrate different components of the framework into reusable experimental workflows.
* `run_methods.py`
  Encapsulates the complete active learning pipeline into callable functions to facilitate experiment reproducibility and large-scale evaluations.
* `start.py`
  Contains the experimental configurations corresponding to the results reported in the manuscript. This script serves as the recommended entry point for reproducing the published experiments.
* `run.py`
  Provides a low-level implementation of the active learning workflow, allowing users to customize the framework and explore additional parameter combinations beyond those considered in this study.
## Installation
Use the following command to create the virtual environment required for this project:
```
conda env create =f environment.yml -n env_name
```

## Usage
### Data Preparation
Before running the active learning pipeline using the datasets described in the manuscript, the corresponding label files and feature files must be downloaded from `to be supplemented` and placed in the root directory of this project.
Please ensure that at least **10 GB of free disk space** is available before starting the experiments.
### Reproducing the Results
To reproduce the active learning (AL) results reported in the paper, run:
```bash
python start.py AL
python start.py AL_save_genes
```
The first command saves only the number of positive hits identified during the experiments, whereas the second command additionally records the identities of the selected samples at each active learning cycle.

To reproduce the one-cycle baseline results reported in the paper, run:

```bash
python start.py one_cycle
python start.py one_cycle_save_genes
```
The first command saves only the number of positive hits, whereas the second command additionally records the identities of the selected samples.
To reproduce the cold-start results reported in the paper, run:
```bash
python start.py cold_start
```
### Multiprocessing Support

To accelerate computation, this project supports multiprocessing. Users may specify the number of worker processes as the final command-line argument. For example:

```bash
python start.py AL 20
```
The example above launches the active learning pipeline using 20 worker processes. For optimal performance, the specified number should not exceed the number of available CPU cores.
Because multiprocessing behavior depends on the operating system and the underlying implementation, we cannot guarantee that this functionality will work consistently across all computing environments.
### Custom Experiments
To explore additional parameter combinations beyond those reported in the manuscript, users may directly modify the parameter configuration section in `run.py`. Detailed descriptions of the available parameters are provided in the corresponding comments.
### License 
All code is under MIT license.
