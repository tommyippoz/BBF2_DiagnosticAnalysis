import random
from pathlib import Path

import numpy
import pandas

from bbf2_lib.AnomalyPredictor import PointWiseAnomalyPredictor, TimeSeriesAnomalyPredictor
from bbf2_lib.Classifier import choose_classifiers
from bbf2_lib.utils import read_config, test_models, print_scores

if __name__ == "__main__":
    """
    This main performs an analysis of CSV files extracted from commaai dataset
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
    all_df = []
    warn_df = []
    no_warn_df = []
    for file in Path(params["csv_files_folder"]).glob("*.csv"):
        try:
            df = pandas.read_csv(file)
            df = df.sort_values(params["time_column"])
            label_obj = df[params["label_column"]].to_numpy()
            data_obj = df.drop(columns=params["columns_to_remove"] + [params["time_column"]] + [params["label_column"]])
            new_obj = {"X": data_obj, "Y": label_obj}
            if sum(df[params["label_column"]]) > 0:
                warn_df.append(new_obj)
                print("\t\tFile Contains warnings")
            else:
                no_warn_df.append(new_obj)
            all_df.append(new_obj)
            if bool(params["verbose"]):
                print(f"Read CSV '{file.name}' of {len(df)} rows")
        except Exception as e:
            print("Error while processing file %s" % file)
    print("Found %d files containing warnings" % len(warn_df))

    # Setting up datasets
    split_index = int(len(warn_df)*float(params["tt_split"]))
    train_sequences = warn_df[:split_index]
    test_sequences = warn_df[split_index:]
    contamination = numpy.average([numpy.average(x["Y"]) for x in train_sequences])

    # Setting Up Analysis Selection of Classifiers
    algs = None
    if params["supervised"]:
        algs = choose_classifiers(params["sup_algs"], contamination=None, verbose=params["verbose"])
    else:
        algs = choose_classifiers(params["uns_algs"], contamination=contamination, verbose=params["verbose"])
    if params["verbose"]:
        print("Exercising %s analysis" % ("SUPERVISED" if params["supervised"] else "UNSUPERVISED"))

    # Choosing the type of Analysis
    predictor = None
    if not params["use_timeseries"]:
        predictor = PointWiseAnomalyPredictor(clf_list=algs, supervised=params["supervised"],
                                              models_folder=params["models_folder"])
    else:
        predictor = TimeSeriesAnomalyPredictor(clf_list=algs, supervised=params["supervised"],
                                              models_folder=params["models_folder"])
    if params["verbose"]:
        print("Exercising %s analysis" % ("TIME-SERIES" if params["use_timeseries"] else "NORMAL"))


    # Fitting Models
    predictor.fit(sequences=train_sequences, verbose=params["verbose"])

    # Testing Models
    test_results = test_models(predictor=predictor, test_sequences=test_sequences, verbose=params["verbose"])
    if params["print_scores"]:
        print_scores(to_print=test_results, analysis_tag="test", output_folder=params["output_folder"], filename=params["scores_filename"])

    # Test Models using data with no warnings
    test_results = test_models(predictor=predictor, test_sequences=no_warn_df, verbose=params["verbose"])
    if params["print_scores"]:
        print_scores(to_print=test_results, analysis_tag="test_nowarn", output_folder=params["output_folder"],
                     filename=params["scores_filename"])