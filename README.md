![Static Badge](https://img.shields.io/badge/python-3.9-blue)　![Static Badge](https://img.shields.io/badge/license-MIT-orange)　![Static Badge](https://img.shields.io/badge/DOI-to%20be%20supplemented-purple)　![Static Badge](https://img.shields.io/badge/Zenodo-to%20be%20supplemented-green)
# AL_for_SL_SR

- [Abstract](#abstract)
- [Modules](#modules)
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)

## Abstract
Synthetic lethality (SL) and synthetic rescue (SR) gene pairs are promising targets for cancer therapy, but their systematic identification is hindered by a vast combinatorial search space and limited samples. Active learning (AL) offers a potential solution for low-data discovery tasks, yet its effectiveness in SL and SR contexts has not been systematically explored. Here, we simulate a low-data scenario, and systematically analyze AL strategies for iterative SL/SR gene-pair prioritization using combinatorial CRISPR double-knockout data. Across the evaluated cell lines, feature representations and machine-learning models, AL generally improves SL/SR discovery compared with non-iterative and random sampling approaches in low-data setting, with more pronounced improvements in SL than in SR. Notably, the acquisition strategy is the major factor contributing to performance variation in certain screening scenarios. Within the evaluated datasets, no single strategy is universally optimal. Exploitation generally performs well in SL, whereas exploration is more effective in SR within the evaluated datasets. Dynamic acquisition, which balances exploration and exploitation, demonstrates competitive performance across both tasks. Overall, these results demonstrate that AL may provide a useful but context-sensitive framework for improving SL/SR gene-pair discovery under limited experimental budgets.

## Modules

The main code is located in `active_learning_prediction/`:

* `active_learning/model.py`
  Implements the predictive models and ensembles used for uncertainty estimation and prediction.
* `active_learning/Handler.py`
  Maintains and updates the labeled training set and unlabeled sample pool throughout each learning cycle.
* `active_learning/experiment.py`
  Integrates model training, sample acquisition, and result recording into a reusable experimental workflow.
* `active_learning/acquisition.py`
  Implements random, exploitation, exploration, BALD, dynamic, similarity, and gene-frequency acquisition strategies.
* `active_learning/get_data.py`
  Loads the prepared input data and initializes the training set and sample pool.
* `run_A549_function_SL_010pct_nstart64.py`
  Defines the experimental configurations and serves as the recommended entry point for running experiments and aggregating results.

## Installation

We recommend creating a fresh Conda environment:

```bash
conda create -n al_sl_sr python=3.9 pip
conda activate al_sl_sr
python -m pip install numpy==1.23.3 scikit-learn==1.0.2 deep-forest==0.1.7 pandas scipy matplotlib tqdm xgboost
```

Install a PyTorch build compatible with Python 3.9, your operating system, and your hardware by following the [official instructions](https://pytorch.org/get-started/locally/). PyTorch is required for importing the framework, including when running CPU-based models.

From the directory containing `active_learning_prediction/`, verify that the dependencies can be imported:

```bash
cd active_learning_prediction
python -c "from active_learning.experiment import Experiment; print('Imports OK')"
```

## Usage

### Data Preparation

The label and feature files are provided in `label_feature/` and are available as a single compressed archive (`label_feature.zip`, approximately 5 MB). Extract the archive before use. Prepare gene-pair input files following the provided example dataset and place them in the same directory as the experiment runner. Input filenames follow the pattern `{cell}_{label_type}_{pct:03d}pct_{feature}.csv`.

A prepared `A549_sensitive_010pct_function.csv` is included for the supplied experiment. Detailed label definitions, feature construction, and data processing are described in the manuscript.

### Running the Experiments

From `active_learning_prediction/`, run:

```bash
python run_A549_function_SL_010pct_nstart64.py
```

The supplied configuration runs the A549 SL experiment using functional features and sensitive labels at the 10% setting. It compares Deep Forest and DNN ensembles with nine acquisition strategies over ten random seeds, starting with 64 labeled pairs and acquiring 32 pairs per cycle for five cycles.

Results are saved automatically under `results_custom/` in configuration-specific directories. They include hit-count histories, selected gene pairs, initialization records, and summaries across seeds. Rerunning the same configuration skips completed runs that pass consistency checks when `RESUME_COMPLETED=True`; incomplete runs restart from initialization.

### CPU and GPU Support

Set `N_JOBS` in the runner to control the parallelism of supported CPU estimators; the default is 8. Experiment combinations run sequentially. DNN models use CUDA when available and otherwise run on the CPU. Adjust resource settings to suit the available hardware.

### Custom Experiments

Modify the configuration section at the top of `run_A549_function_SL_010pct_nstart64.py` to select cell lines, feature types, SL/SR tasks, models, acquisition strategies, seeds, initial training sizes, batch size, and cycle count. Prepare matching input files for every selected dataset. The runner uses the full remaining candidate pool without reserving a held-out test set.

## License

All code is under the MIT license.
