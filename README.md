# SQUEzE

Spectroscopic QUasar Extractor and redshift (z) Estimator

requirements:
* python 2.7 or 3.6
* argparse
* numpy
* pandas
* sklearn - only for training
* astropy

## Description

SQUEzE is a software package to identify quasars and estimate their redshift in a sample of spectra.
The quasars id and redshif is estimated in a two-step process:
    1. Generate a high completeness and low purity sample of candidates
    2. Filter the candidates to improve purity while keeping a target completeness
See Perez-Rafols et al. 2019 (https://arxiv.org/abs/1903.00023) for more details. Consider referencing
this paper if you are using SQUEzE in your analysis.

SQUEzE can run in different modes:
    1. training - Use a known sample to decide on model options
    2. test - Use a known sample to assess model performance
    3. operation - Apply cuts to unknown sample to build a catalogue
    4. merge - Merge different candidates objects into a single candidate objects

## Installation

download
```
git clone https://github.com/iprafols/SQUEzE.git
```
Add <path_to_SQUEzE>/`SQUEzE/py/` to your `PYTHONPATH`.

It is recommended to pretrain SQUEzE using the provided results from BOSS.
To do so run
```
cd <path_to_SQUEzE>/SQUEzE/data
tar -xJf BOSS_train_64plates.tar.xz
cd ../bin
python pretrain_squeze.py
```
from the `bin/` folder. This will generate a trained model named `BOSS_train_64plates_model.json`
stored in the `data/` folder.


## Usage

SQUEzE presents four modes of operation: training, test, operation, and merge. The training
and test modes are used on a controlled sample where a truth table is available,
and allows the code to learn or test performance of the model parameters.
The operation mode is used on an uncontrolled sample to generate a quasar catalogue.
The merge mode is used to join different (and possibly parallel) runs on a different mode.

### Formatting data (training, test, operation, and candidates modes)

Before running SQUEzE data must be formatted so that the code can use it properly.
This section explains both the optional and required pre-steps to format all data
correctly.

1. Formatting spectra (required):

The `spectra` variable sets the spectra where SQUEzE will be run. Format of
the spectra must be a Spectra object (defined in py/squeze_spectra.py) containing
a set of Spectrum instances. The package provides the file `format_superset_dr12q.py´
with a working example of how to format BOSS spectra.

2.1 Formatting lines (optional):

The `lines` variable sets the characteristics of the lines used by the code.
For SQUEzE to use a value different from the default, it needs to be
formatted as a pandas data frame and saved into a json file.
The package provides the file `format_lines.py´ with the instructions to
properly create this object. It is a modifyiable working example.

### Usage in training mode

This mode is thought to create models. Data with labels is required to be able to train the models. Candidates can either be loaded or be looked for prior to the model training.

Run
```
python squeze_training.py
```
for an explanation on the arguments add `-h` to the previous line

### Usage in test mode

This mode is thought to test the performance of the models. Data with labels is required. Candidates can either be loaded or be looked for prior to the model testing.

Run
```
python squeze_test.py
```
for an explanation on the arguments add `-h` to the previous line

### Usage in operation mode

This mode is thought to run SQUEzE during normal operations. Data is expected not to have labels. Candidates can either be loaded or be looked for prior to the model prediction.

Run
```
python squeze_operation.py
```
for an explanation on the arguments add `-h` to the previous line

### Usage in candidates mode

This mode is thought to only look for candidates. For large samples, it might help speed up the proces by breaking it in smaller bits. Those bits can be later merged using the merge mode. Data is  not required to have labels at this stage, but it is if it is going to be used for training or testing .

Run
```
python squeze_candidates.py
```
for an explanation on the arguments add `-h` to the previous line

### Usage in merge mode

This mode is though to be used to merge different candidate files.

Run
```
python squeze_merge_candidates.py
```
for an explanation on the arguments add `-h` to the previous line
