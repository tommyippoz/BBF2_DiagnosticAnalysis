import copy
import os
import random
from pathlib import Path

import now
import numpy
import pandas

from bbf2_lib import AnomalyPredictor
from bbf2_lib.AnomalyPredictor import PointWiseAnomalyPredictor, TimeSeriesAnomalyPredictor
from bbf2_lib.Classifier import choose_classifiers, get_classifier_name
from bbf2_lib.predictor_utils import test_models, store_models
from bbf2_lib.utils import read_config, print_scores

if __name__ == "__main__":
    """
    This main performs an analysis of CSV files extracted from commaai dataset, assuming models are already learned
    Parameter configuration via "debug_cfg.cfg" file
    """

    # Reads params file
    params = read_config("debug_cfg.cfg")
    if params is None:
        print("Cannot read CFG file. Please provide a debug_cfg.cfg file, same folder of this script.")
        exit(1)

    # Sets random seed to increase repeatability
    random.seed(int(params["rng_seed"]))
    numpy.random.seed(int(params["rng_seed"]))
    random_state = int(params["rng_state"])

    # Reading CSVs as DataFrames
    test_sequences = []
    for file in Path(params["csv_files_folder"]).glob("*.csv"):
        try:
            df = pandas.read_csv(file)
            df = df.sort_values(params["time_column"])
            if sum(df[params["label_column"]]) > 0:
                df.to_csv(os.path.join(params["test_csv_folder"], file.name), index=False)
            if bool(params["verbose"]):
                print(f"Read CSV '{file.name}' of {len(df)} rows")
        except Exception as e:
            print("Error while processing file %s" % file)
    print("Found %d test files" % len(test_sequences))