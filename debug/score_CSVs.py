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
        print("Cannot read CFG file. Please provide a debug_cfg.cfg file in the same folder of this script.")
        exit(1)

    # Sets random seed to increase repeatability
    random.seed(int(params["rng_seed"]))
    numpy.random.seed(int(params["rng_seed"]))
    random_state = int(params["rng_state"])

    # Reading CSVs as DataFrames
    test_sequences = []
    for file in Path(params["test_csv_folder"]).glob("*.csv"):
        try:
            df = pandas.read_csv(file)
            df = df.sort_values(params["time_column"])
            label_obj = df[params["label_column"]].to_numpy()
            data_obj = df.drop(columns=params["columns_to_remove"] + [params["time_column"]] + [params["label_column"]])
            test_sequences.append({"X": data_obj, "Y": label_obj, "filename": file.name})
            if bool(params["verbose"]):
                print(f"Read CSV '{file.name}' of {len(df)} rows")
        except Exception as e:
            print("Error while processing file %s" % file)
    print("Found %d test files" % len(test_sequences))

    # Setting Up Analysis Selection of Classifiers
    algs = None
    if params["supervised"]:
        algs = choose_classifiers(params["sup_algs"], contamination=None, verbose=False)
    else:
        algs = choose_classifiers(params["uns_algs"], contamination=0.1, verbose=False)
    if params["verbose"]:
        print("Loading existing classifiers for %s analysis" % ("SUPERVISED" if params["supervised"] else "UNSUPERVISED"))

    # Choosing the type of Analysis
    predictor = AnomalyPredictor.load(params["models_folder"], params["use_timeseries"],
                                      params["supervised"], algs, params["verbose"])

    # Testing Models
    for csv_data in test_sequences:
        test_csv = [csv_data]
        test_results, predictions = test_models(predictor=predictor, test_sequences=test_csv, verbose=params["verbose"])
        print_scores(to_print=test_results, analysis_tag="test",
                     output_folder=params["out_dataframes_folder"], filename=params["scores_filename"])
        if params["print_predictions"]:
            # Prep CSV
            to_print = copy.deepcopy(csv_data["X"])
            to_print[params["label_column"]] = csv_data["Y"]
            for i in range(0, len(predictor.clf_list)):
                to_print["[PRED]" + get_classifier_name(predictor.clf_list[i])] = predictions[i]
            to_print.to_csv(os.path.join(params["out_dataframes_folder"].replace('"', ''), csv_data["filename"].replace(".csv", "_PREDICTED.csv")), index=False)


