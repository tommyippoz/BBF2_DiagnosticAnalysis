import copy
import os.path

import joblib
import numpy
import pandas

from bbf2_lib.Classifier import get_classifier_name
from debug.test_unsupervised import current_ms

# -------------------- UTILITY FUNCTIONS ----------------------------
def load(models_folder, use_timeseries, supervised, algs, verbose):
    clf_list = []
    for alg in algs:
        clf_name = get_classifier_name(alg)
        filename = os.path.join(models_folder,
                                "timeseries" if use_timeseries else "point",
                                clf_name,
                                "model.joblib")
        if os.path.exists(filename):
            clf_model = joblib.load(filename)
            if verbose:
                print("\tLoaded '%s' model" % get_classifier_name(clf_model))
            clf_list.append(clf_model)

    if not use_timeseries:
        predictor = PointWiseAnomalyPredictor(clf_list, supervised, models_folder)
    else:
        predictor = TimeSeriesAnomalyPredictor(clf_list, supervised, models_folder)

    return predictor

class AnomalyPredictor:
    """
    Class to manage the analysis of CSV files and predict label
    """

    def __init__(self, clf_list: list, supervised: bool = True, models_folder: str = "./models"):
        """
        Constructor
        :param clf_list: list of classifiers to be compares
        :param models_folder: folder where fitted models are stored
        :param supervised: True if the analysis has to be supervised
        """
        self.clf_list = clf_list
        self.supervised = supervised
        self.models_folder = models_folder

    def fit(self, sequences: list, verbose:bool=True):
        """
        Trains classifiers according to the setup
        :param verbose: True if debug information to be shown
        :param sequences: CSV data, to be partitioned for fitting
        :return:
        """
        x_train = self.extract_data(sequences)
        y_train = numpy.concatenate([item["Y"] for item in sequences], axis=0)
        if verbose:
            print("Train data and labels created: %d items" % len(y_train))
        for clf in self.clf_list:
            start_ms = current_ms()
            if self.supervised:
                clf.fit(x_train, y_train)
            else:
                clf.fit(x_train)
            end_ms = current_ms()
            if verbose:
                print("\tTraining of classifier %s ended: %d ms" % (get_classifier_name(clf), (end_ms - start_ms)))
        return self

    def extract_data(self, sequences: list):
        """
        TO BE OVERRIDDEN
        Trains classifiers according to the setup
        :param sequences: CSV data, to be partitioned for fitting
        :return:
        """
        return None, None

    def predict(self, sequences: list, verbose:bool = True) -> list:
        """
        Predicts using classifiers according to the setup
        :param verbose: True if debug information to be shown
        :param sequences: CSV data, to be partitioned for testing
        :return: list of dictionaries containing clf_name, predictions, time needed to predict
        """
        x_test = self.extract_data(sequences)
        results = []
        predictions = []
        for clf in self.clf_list:
            start_ms = current_ms()
            pred_label = clf.predict(x_test)
            end_ms = current_ms()
            predictions.append(pred_label)
            results.append({"clf": get_classifier_name(clf),
                            "model": clf,
                            "predictions": pred_label,
                            "predict_time": end_ms - start_ms,
                            "predict_time_per_item": (end_ms - start_ms)/len(x_test)})
            if verbose:
                print("\tClassifier %s exercised in %d ms" % (results[-1]["clf"], results[-1]["predict_time"]))
        return results, predictions


class PointWiseAnomalyPredictor(AnomalyPredictor):

    def __init__(self, clf_list: list, supervised: bool = True, models_folder: str = None):
        super().__init__(clf_list, supervised, models_folder)

    def extract_data(self, sequences: list) -> numpy.ndarray:
        """
        Trains classifiers according to the setup
        :param sequences: CSV data, to be partitioned for fitting
        :return:
        """
        x_data = pandas.concat([item["X"] for item in sequences]).to_numpy()
        return x_data


class TimeSeriesAnomalyPredictor(AnomalyPredictor):

    def __init__(self, clf_list: list, supervised: bool = True, models_folder: str = None):
        super().__init__(clf_list, supervised, models_folder)

    def extract_data(self, sequences: list):
        """
        Trains classifiers according to the setup
        :param sequences: CSV data, to be partitioned for fitting
        :return:
        """
        x_data = pandas.concat([self.add_time_features(item["X"]) for item in sequences]).to_numpy()
        return x_data

    def add_time_features(self, in_df: pandas.DataFrame) -> pandas.DataFrame:
        """
        Creates additional features starting from the consumption feature
        :return: nothing
        """
        # Init DataFrame
        new_f = copy.deepcopy(in_df)
        for f_name in in_df.columns:
            
            new_f[f_name + ' [diff t-1]'] = in_df[f_name] - in_df[f_name].shift(1)
            new_f[f_name + ' [diff t-2]'] = in_df[f_name] - in_df[f_name].shift(2)
            new_f[f_name + ' [diff t-5]'] = in_df[f_name] - in_df[f_name].shift(5)
            new_f[f_name + ' [diff t-10]'] = in_df[f_name] - in_df[f_name].shift(10)
            new_f = new_f.fillna(0)
    
            #  Relative Differences between Features
            new_f[f_name + ' [rdiff t-1]'] = new_f[f_name + ' [diff t-1]'] / in_df[f_name]
            new_f[f_name + ' [rdiff t-2]'] = new_f[f_name + ' [diff t-2]'] / in_df[f_name]
            new_f[f_name + ' [rdiff t-5]'] = new_f[f_name + ' [diff t-5]'] / in_df[f_name]
            new_f[f_name + ' [rdiff t-10]'] = new_f[f_name + ' [diff t-10]'] / in_df[f_name]
            new_f = new_f.fillna(1)
            new_f = new_f.replace([numpy.inf, -numpy.inf], 0)
    
            # Moving Averages
            new_f[f_name + ' [diff ma-2]'] = in_df[f_name] - in_df.rolling(window=2)[f_name].mean()
            new_f[f_name + ' [diff ma-5]'] = in_df[f_name] - in_df.rolling(window=5)[f_name].mean()
            new_f[f_name + ' [diff ma-10]'] = in_df[f_name] - in_df.rolling(window=10)[f_name].mean()
            new_f = new_f.fillna(0)

        return new_f
