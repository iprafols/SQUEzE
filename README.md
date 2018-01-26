# SQUEzE

Spectroscopic QUasar Extractor and redshift (z) Estimator

requirements:
* python 2.7
* argparse
* numpy
* pandas
* scipy
* astropy
* tqdm
* pickle

## Description

SQUEzE is a software package to identify quasars and estimate their redshift in a sample of spectra.
The quasars id and redshif is estimated in a two-step process:
    1. Generate a high completeness and low purity sample of candidates
    2. Filter the candidates to improve purity while keeping a target completeness
    
SQUEzE can run in different modes:
    1. training - Use a known sample to decide on which cuts to apply
    2. operation - Apply cuts to unknown sample to build a catalogue
    3. merge - Merge different candidates objects into a single candidate objects

## Installation

download
```
git clone https://github.com/iprafols/SQUEzE.git
```

run
```
python setup.py install --user
```
(assuming you run as user; for a system-wide install omit `--user` option).

Alternatively, you can just add `SQUEzE/py/` to your `PYTHONPATH`.

## Usage

SQUEzE presents two modes of operation: training and operation. The training mode
is used on a controlled sample where a truth table is available, and allows the code to
learn the cuts required to achieve the target completeness. The operation mode is uesd
on an uncontrolled sample to generate a quasar catalogue.

### Formatting data (training and operation modes)

Before running SQUEzE data must be formatted so that the code can use it properly.
This section explains both the optional and required pre-steps to format all data
correctly.

1. Formatting spectra (required):

The `spectra` variable sets the spectra where SQUEzE will be run. Format of
the spectra must be a Spectra object (defined in py/squeze_spectra.py) containing
a set of Spectrum instances. The package provides the file `format_boss_spectra.py´
with a working example of how to format BOSS spectra. 

2. Formatting lines (optional):

The `lines` variable sets the characteristics of the lines used by the code.
For SQUEzE to use a value different from the default, it needs to be
formatted as a pandas data frame and saved into a pkl file (python binary).
The package provides the file `format_lines.py´ with the instructions to
properly create this object. It is a modifyiable working example.

3. Formatting cuts (optional):

The `cuts` variable sets the characteristics of the cuts on the candidates to
be used by the code.
For SQUEzE to use a value different from the default, it needs to be
formatted as a list with tuples (name, value, type) and  be saved into a pkl file
(python binary).
The package provides the file `format_cuts.py´ with the instructions to
properly create this object. It is a modifyiable working example.

### Usage in training mode

run
```
python squeze_training.py
```
options:
* --qso-cat : Name of the fits file containig the quasar catalogue
* --qso-cols : White-spaced list of the data arrays (of the quasar catalogue) to be loaded
* --qso-hdu : Number of the Header Data Unit in --qso-cat where the catalogue is stored
* --qso-specid : Name of the column that will be used as specid. Must be included in --qso-col
* --input-spectra : Name of the pkl file(s) containing the spectra that are to be analysed. If multiple files are given, then they must be passed as a white-spaced list. The spectra in each of the files will be loaded into memory as a block, and candidates will be looked for before loading the next set of spectra.
* --load-candidates : Load candidates previously found. If --input-candidates is passed, then load from there. Otherwise, load from --output-candidates.
* --lines : Name of the pkl file containing the lines ratios to be computed
* --cuts : Name of the pkl file containing the cuts to be applied
* --z-precision : Maximum difference betwee the true redshift and the measured redshift for a candidate to be considered a true detection
* --input-candidates : Name of the pkl file from where candidates will be loaded
* --output-candidates : Name of the pkl file where the candidates will be saved
* --output-cuts : Name of the pkl file where the cuts will be saved
* --quiet : Do not print messages

### Usage in operation mode


### Usage in merging mode

run
```
python squeze_merge_candidates.py
```
options:
* --input-candidates : List of pkl files containing candidates objects to merge.
* --output-candidates : Name of the pkl file where the candidates will be saved
