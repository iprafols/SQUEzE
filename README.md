# SQUEzE

Spectroscopic QUasar Extractor and redshift (z) Estimator

requirements:
* python 2.7 or 3.6
* argparse
* numpy
* pandas
* scipy
* sklearn
* astropy
* tqdm
* pickle

## Description

SQUEzE is a software package to identify quasars and estimate their redshift in a sample of spectra.
The quasars id and redshif is estimated in a two-step process:
    1. Generate a high completeness and low purity sample of candidates
    2. Filter the candidates to improve purity while keeping a target completeness
See Perez-Rafols et al. 2019 (https://arxiv.org/abs/1903.00023) for more details. Consider referencing
this paper if you are using SQUEzE in your analysis.
    
SQUEzE can run in different modes:
    1. training - Use a known sample to decide on which cuts to apply
    2. operation - Apply cuts to unknown sample to build a catalogue
    3. merge - Merge different candidates objects into a single candidate objects

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
tar -xzvf BOSS_train_64plates.tar.gz
cd ../bin
python pretrain_squeze.py
```
from the `bin/` folder. This will generate a model named `BOSS_train_64plates_model.pkl`
stored in the `data/` folder.


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

2.1 Formatting lines (optional):

The `lines` variable sets the characteristics of the lines used by the code.
For SQUEzE to use a value different from the default, it needs to be
formatted as a pandas data frame and saved into a pkl file (python binary).
The package provides the file `format_lines.py´ with the instructions to
properly create this object. It is a modifyiable working example.

### Usage in training mode

run
```
python squeze_training.py
```
optional arguments:

-h, --help            
Show this help message and exit

--quiet               
Do not print messages (default: False)

--input-spectra INPUT_SPECTRA [INPUT_SPECTRA ...]               
Name of the pkl file(s) containing the spectra that are to be analysed. If multiple files 
are given, then they must be passed as a white-spaced list. The spectra in each of the 
files will be loaded into memory as a block, and candidates will be looked for before 
loading the next set of spectra. (default: None)

--load-candidates                    
Load candidates previously found. If --input- candidates is passed, then load from there. 
Otherwise, load from --output-candidates. (default: False)

--input-candidates INPUT_CANDIDATES                 
Name of the pkl file from where candidates will be loaded. (default: None)

--output-candidates OUTPUT_CANDIDATES               
Name of the pkl file where the candidates will be saved. In training mode, the model will 
be saved using this name (without the extension) as base name and append the extension 
_model.pkl to it (default: None)

--check-statistics                  
Check the candidates' statistics at the end (default: False)

--check-probs CHECK_PROBS [CHECK_PROBS ...]               
White-spaced list of the probabilities to check. The candidates' statistics will be computed 
for these cuts in probabilities. Ignored if --check-statistics is not passed. If it is not passed 
and --check-statistics is then np.arange(0.9, 0.0, -0.05) (default: None)

--save-fits                          
Save the final catalogue also as a fits file (default: False)

--peakfind-width PEAKFIND_WIDTH               
Width (in pixels) of the tipical peak (default: None)

--peakfind-sig PEAKFIND_SIG               
Minimum significance required to accept a peak (default: None)

--qso-dataframe QSO_DATAFRAME               
Name of the pkl file containing the quasar catalogue formatted into pandas dataframe. Must 
only contain information of quasars that will be loaded. Must be present if --qso-cat is not 
passed. (default: None)

--qso-cat QSO_CAT                    
Name of the fits file containig the quasar catalogue. Must be present if --qso-dataframe is not 
passed (default: None)

--qso-cols QSO_COLS [QSO_COLS ...]               
White-spaced list of the data arrays (of the quasar catalogue) to be loaded. Must be present 
only if --qso-cat is passed (default: None)

--qso-hdu QSO_HDU                    
Number of the Header Data Unit in --qso-cat where the catalogue is stored. (default: 1)

--qso-specid QSO_SPECID               
Name of the column that will be used as specid. Must be included in --qso-cols. Must be present 
only if --qso-cat is passed (default: None)

--z-precision Z_PRECISION               
Maximum difference betwee the true redshift and the measured redshift for a candidate to be 
considered a true detection. This option only works on cuts of type 'percentile'. (default: None)

--lines LINES                        
Name of the pkl file containing the lines ratios to be computed. (default: None)

--cuts CUTS                          
Name of the pkl file containing the hard-core cuts to be included in the model. (default: None)

--try-lines [TRY_LINES [TRY_LINES ...]]               
Name of the lines that will be associated to the peaks to estimate the redshift. (default: None)

--weighting-mode WEIGHTING_MODE               
Selects the weighting mode when computing the line ratios. Can be 'weights' if ivar is to be 
used as weights when computing the line ratios, 'flags' if ivar is to be used as flags when 
computing the line ratios (pixels with 0 value will be ignored, the rest will be averaged without 
weighting), or 'none' if weights are to be ignored. (default: weights)

### Usage in test mode

run
```
python squeze_test.py
```
optional arguments:

-h, --help            
Show this help message and exit

--quiet               
Do not print messages (default: False)

--input-spectra INPUT_SPECTRA [INPUT_SPECTRA ...]               
Name of the pkl file(s) containing the spectra that are to be analysed. If multiple files are given, 
then they must be passed as a white-spaced list. The spectra in each of the files will be loaded 
into memory as a block, and candidates will be looked for before loading the next set of spectra.
(default: None)

--load-candidates                    
Load candidates previously found. If --input-candidates is passed, then load from there. Otherwise,
load from --output-candidates. (default: False) 

--input-candidates INPUT_CANDIDATES                
Name of the pkl file from where candidates will be loaded. (default: None)

--output-candidates OUTPUT_CANDIDATES               
Name of the pkl file where the candidates will be saved. In training mode, the model will be saved 
using this name (without the extension) as base name and append the extension _model.pkl to it 
(default: None)

--check-statistics                  
Check the candidates' statistics at the end (default: False)

--check-probs CHECK_PROBS [CHECK_PROBS ...]               
White-spaced list of the probabilities to check. The candidates' statistics will be computed for 
these cuts in probabilities. Ignored if --check-statistics is not passed. If it is not passed and 
--check-statistics is then np.arange(0.9, 0.0, -0.05) (default: None)

--save-fits           
Save the final catalogue also as a fits file (default: False)

--qso-dataframe QSO_DATAFRAME               
Name of the pkl file containing the quasar catalogue formatted into pandas dataframe. Must only 
contain information of quasars that will be loaded. Must be present if --qso-cat is not passed. 
(default: None)

--qso-cat QSO_CAT                    
Name of the fits file containig the quasar catalogue. Must be present if --qso-dataframe is not 
passed (default: None)

--qso-cols QSO_COLS [QSO_COLS ...]               
White-spaced list of the data arrays (of the quasar catalogue) to be loaded. Must be present only 
if --qso-cat is passed (default: None)

--qso-hdu QSO_HDU                    
Number of the Header Data Unit in --qso-cat where the catalogue is stored. (default: 1)

--qso-specid QSO_SPECID               
Name of the column that will be used as specid. Must be included in --qso-cols. Must be present 
only if --qso-cat is passed (default: None)

--model MODEL                        
Name of the pkl file containing the model to be used in the computation of the probabilities of 
candidates being quasars (default: None)

### Usage in operation mode

run
```
python squeze_operation.py
```
optional arguments:

-h, --help           
Show this help message and exit

--quiet               
Do not print messages (default: False)

--input-spectra INPUT_SPECTRA [INPUT_SPECTRA ...]               
Name of the pkl file(s) containing the spectra that are to be analysed. If multiple files are given, then 
they must be passed as a white-spaced list. The spectra in each of the files will be loaded into memory 
as a block, and candidates will be looked for before loading the next set of spectra. (default: None)

--load-candidates                    
Load candidates previously found. If --input-candidates is passed, then load from there. Otherwise,
load from --output-candidates. (default: False)

--input-candidates INPUT_CANDIDATES               
Name of the pkl file from where candidates will be loaded. (default: None)

--output-candidates OUTPUT_CANDIDATES               
Name of the pkl file where the candidates will be saved. In training mode, the model will be saved 
using this name (without the extension) as base name and append the extension _model.pkl to it 
(default: None)

--check-statistics                   
Check the candidates' statistics at the end (default:False)

--check-probs CHECK_PROBS [CHECK_PROBS ...]               
White-spaced list of the probabilities to check. The candidates' statistics will be computed for 
these cuts in probabilities. Ignored if --check-statistics is not passed. If it is not passed and 
--check-statistics is then np.arange(0.9, 0.0, -0.05) (default: None)

--save-fits                          
Save the final catalogue also as a fits file (default: False)

--output-catalogue OUTPUT_CATALOGUE               
Name of the fits file where the final catalogue will be stored. (default: None)

--prob-cut PROB_CUT                  
Only objects with probability > PROB_CUT will be included in the catalogue (default: 0.1)

### Usage in merging mode

run
```
python squeze_merge_candidates.py
```
optional arguments:
-h, --help                 
Show this help message and exit

--quiet                     
Do not print messages (default: False)

--input-candidates INPUT_CANDIDATES [INPUT_CANDIDATES ...]               
List of pkl files containing candidates objects to merge. (default: None)

--output-candidates OUTPUT_CANDIDATES               
Name of the pkl file where the candidates will be saved. (default: None)
