# BBF2_DiagnosticAnalysis
Repository containing analyses of Vehicle Diagnostics Data
Collaboration between SPINDOX, UNIFI, Resiltech

## Data
Data is fetched from the commaCarSegments - commaai dataset (here -> https://huggingface.co/datasets/commaai/commaCarSegments)
This test is conducted using CSV files extracted using the SUBARU_OUTBACK tag
The interesting part is a feature 'Steer_Warning', which is the objective of the analysis (the one to predict)
Other features:
- CHECKSUM
- COUNTER
- Steer_Error_1: seems always 0, not used in the analysis
- Steer_Torque_Sensor
- Steer_Error_2: not used in the analysis, seems an additional label
- Steer_Warning: the feature to predict
- Steering_Angle
- Steer_Torque_Output
- time: timestamp used to sort rows - not used for analysis
- log_name: str name of the log, constant across a single file
- addr: seems always 281?
- fileID: str name of the file, constant across a single file
- wordcount
- name: name derived from CAN flow, seems always constant

Overall, features used to predict the 'Steer_Warning' feature are:
- CHECKSUM
- COUNTER
- Steer_Torque_Sensor
- Steering_Angle
- Steer_Torque_Output
- wordcount

ZIP Containing the full CSV list can be downloaded at the link [here](https://drive.google.com/file/d/1gzZaWqarOESrZeEKsWBCfta9E42tqvhz/view?usp=sharing)

## How to run the Code
The code is available in the repository. Steps to execute it:
1) clone the repo
2) download the ZIP above and unzip it in the root folder of the repo you downloaded
3) create your new Python venv. For testing, Python 3.12 was used, package list available [here](pip_freeze.txt)

### Training Prediction Models
4) take a look at the [debug](debug) folder, where 2 things are super important:
- 4.1) a .cfg file containing the setup of the analysis
- 4.2) a [file](debug/learn_anomaly_detectors.py), which reads the CFG file above, learns and tests prediction models for the 'Steer_Warning' variable
- 4.3) if the "store_models" variable is set to True in the cfg file, the models will be stored for future use without needing re-training

### Exercising Prediction Models
Once step 4) is executed and models are stored in the filesystem, they could be loaded at will for analyses.

5) The [file](debug/score_allalgs_CSVs.py) allows for predicting the target (Steer_Warning) variable for one or more CSVs stored in a folder, using all the available models resulting from different step 4) iterations. The predictions are printed as additional columns to the CSVs and stored in a dedicated output folder for further analysis

## State of the Tests
Tommaso -> at the moment I am finalizing some tests. TODOList includes
- saving learned models to file so that they could be loaded and used without needing re-training
- making an additional script that takes a new CSV file and a saved model and provides you with predictions of that model for each row of that file
- check stability
- making a report of the "anoalies I find"
