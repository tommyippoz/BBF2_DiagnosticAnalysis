import copy
import os.path

import joblib
import numpy
import pandas
from pyod.models.base import BaseDetector

from bbf2_lib.Classifier import get_classifier_name
from bbf2_lib.PredictionExplainer import SHAPExplainer
from debug.test_unsupervised import current_ms

# -------------------- UTILITY FUNCTIONS ----------------------------
def load(models_folder, use_timeseries, supervised, algs, verbose):
    """
    To be used to create an AnomalyPredictor from an existing block of models
    :param models_folder: folder to read
    :param use_timeseries: Truye if uses timeseries
    :param supervised: True if supervised ALgorithms
    :param algs: list of algorithms
    :param verbose: True if debug information to be shown
    :return:
    """
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

def load_all(models_folder:str, verbose:bool = True):
    """
    To be used to create an AnomalyPredictor from an existing block of models
    :param models_folder: folder to read
    :param verbose: True if debug information to be shown
    :return:
    """
    t_clf_list = []
    t_explainers = {}
    if os.path.exists(os.path.join(models_folder, "timeseries")):
        for sub in os.listdir(os.path.join(models_folder, "timeseries")):
            if os.path.isdir(os.path.join(models_folder, "timeseries", sub)):
                filename = os.path.join(models_folder, "timeseries", sub, "model.joblib")
                if os.path.exists(filename):
                    clf_model = joblib.load(filename)
                    if verbose:
                        print("\tLoaded '%s' model" % get_classifier_name(clf_model))
                    t_clf_list.append(clf_model)
                    t_explainers[get_classifier_name(clf_model)] = {}
                    exp_fn = os.path.join(models_folder, "timeseries", sub, "shap_explainer.joblib")
                    if os.path.exists(exp_fn):
                        t_explainers[get_classifier_name(clf_model)]["SHAP"] = joblib.load(exp_fn)

    p_clf_list = []
    p_explainers = {}
    if os.path.exists(os.path.join(models_folder, "point")):
        for sub in os.listdir(os.path.join(models_folder, "point")):
            if os.path.isdir(os.path.join(models_folder, "point", sub)):
                filename = os.path.join(models_folder, "point", sub, "model.joblib")
                if os.path.exists(filename):
                    clf_model = joblib.load(filename)
                    if verbose:
                        print("\tLoaded '%s' model" % get_classifier_name(clf_model))
                    p_clf_list.append(clf_model)
                    p_explainers[get_classifier_name(clf_model)] = {}
                    exp_fn = os.path.join(models_folder, "point", sub, "shap_explainer.joblib")
                    if os.path.exists(exp_fn):
                        p_explainers[get_classifier_name(clf_model)]["SHAP"] = joblib.load(exp_fn)

    predictor = AnomalyPredictorBunch(ap_list=
        [PointWiseAnomalyPredictor(clf_list=p_clf_list, models_folder=models_folder, explainers=p_explainers),
                TimeSeriesAnomalyPredictor(clf_list=t_clf_list, models_folder=models_folder, explainers=t_explainers)],
                                      models_folder=models_folder)

    return predictor

class AnomalyPredictor:
    """
    Class to manage the analysis of CSV files and predict label
    """

    def __init__(self, clf_list: list, supervised: bool = True, models_folder: str = "./models",
                 shap_explain: bool = True, n_explanations:int = 1000, explainers: dict = None):
        """
        Constructor
        :param shap_explain: true if wants to build model that explains via SHAP
        :param clf_list: list of classifiers to be compares
        :param models_folder: folder where fitted models are stored
        :param supervised: True if the analysis has to be supervised
        """
        self.clf_list = clf_list
        self.supervised = supervised
        self.models_folder = models_folder
        self.shap_explain = shap_explain
        self.n_explanations = n_explanations
        self.explainers = explainers

    def fit(self, sequences: list, verbose:bool=True):
        """
        Trains classifiers according to the setup
        :param verbose: True if debug information to be shown
        :param sequences: CSV data, to be partitioned for fitting
        :return:
        """
        if self.clf_list is not None and isinstance(self.clf_list, list) and len(self.clf_list) > 0:
            x_train = self.extract_data(sequences)
            y_train = numpy.concatenate([item["Y"] for item in sequences], axis=0)
            if verbose:
                print("Train data and labels created: %d items" % len(y_train))
            self.explainers = {get_classifier_name(clf): None for clf in self.clf_list}
            for clf in self.clf_list:
                # Trains Classifier
                start_ms = current_ms()
                if self.supervised:
                    clf.fit(x_train, y_train)
                else:
                    clf.fit(x_train)
                end_ms = current_ms()
                if verbose:
                    print("\tTraining of classifier %s ended: %d ms" % (get_classifier_name(clf), (end_ms - start_ms)))
                # Trains Explainers
                exp_dict = {}
                train_expl = x_train[0:self.n_explanations, :]
                if self.shap_explain:
                    start_ms = current_ms()
                    s_exp = SHAPExplainer(clf)
                    s_exp.train(train_expl, list(sequences[0]["X"]))
                    exp_dict["SHAP"] = s_exp
                    if verbose:
                        print("\t\tSHAP Model learned in %d ms using %d train data points" %
                              (current_ms() - start_ms, self.n_explanations))

                self.explainers[get_classifier_name(clf)] = exp_dict

        else:
            print("Parameter clf_list is wrong, cannot train AnomalyPredictor")
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
                            "use_timeseries": isinstance(self, TimeSeriesAnomalyPredictor),
                            "is_supervised": not isinstance(clf, BaseDetector),
                            "predictions": pred_label,
                            "predict_time": end_ms - start_ms,
                            "predict_time_per_item": (end_ms - start_ms)/len(x_test)})
            if verbose:
                print("\tClassifier %s exercised in %d ms" % (results[-1]["clf"], results[-1]["predict_time"]))
        return results, predictions

    def explain(self, sequences: list, verbose:bool = True) -> dict:
        """
        Predicts using classifiers according to the setup
        :param verbose: True if debug information to be shown
        :param sequences: CSV data, to be partitioned for testing
        :return: dict of dictionaries with key=clf_name, value=dict(key=SHAP, value=explanations)
        """
        x_test = self.extract_data(sequences)
        explanations = {}
        if verbose:
            print("\nComputing Explanations for %d items..." % x_test.shape[0])
        for clf in self.clf_list:
            clf_name = get_classifier_name(clf)
            if "SHAP" in self.explainers[clf_name]:
                start_ms = current_ms()
                explanations["SHAP"] = self.explainers[clf_name]["SHAP"].explain(x_test)
                if verbose:
                    print("\tSHAP Explanations for %s derived in %d ms" % (clf_name, current_ms() - start_ms))

        return explanations


class PointWiseAnomalyPredictor(AnomalyPredictor):

    def __init__(self, clf_list: list, supervised: bool = True, models_folder: str = None,
                 shap_explain: bool = True, n_explanations:int = 1000, explainers: dict = None):
        super().__init__(clf_list, supervised, models_folder, shap_explain, n_explanations, explainers)

    def extract_data(self, sequences: list) -> numpy.ndarray:
        """
        Trains classifiers according to the setup
        :param sequences: CSV data, to be partitioned for fitting
        :return:
        """
        x_data = pandas.concat([item["X"] for item in sequences]).to_numpy()
        return x_data


class TimeSeriesAnomalyPredictor(AnomalyPredictor):

    def __init__(self, clf_list: list, supervised: bool = True, models_folder: str = None,
                 shap_explain: bool = True, n_explanations:int = 1000, explainers: dict = None):
        super().__init__(clf_list, supervised, models_folder, shap_explain, n_explanations, explainers)

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

class AnomalyPredictorBunch(AnomalyPredictor):

    def __init__(self, ap_list: list, supervised: bool = True, models_folder: str = None):
        super().__init__(None, supervised, models_folder)
        self.ap_list = ap_list
        self.clf_list = [x for ap in ap_list for x in ap.clf_list]
        self.is_ts_list = [isinstance(ap, TimeSeriesAnomalyPredictor) for ap in ap_list for x in ap.clf_list]

    def fit(self, sequences: list, verbose:bool=True):
        """
        Trains classifiers according to the setup
        :param verbose: True if debug information to be shown
        :param sequences: CSV data, to be partitioned for fitting
        :return:
        """
        for ap in self.ap_list:
            ap.fit(sequences, verbose)
        return self


    def predict(self, sequences: list, verbose:bool = True) -> list:
        """
        Predicts using classifiers according to the setup
        :param verbose: True if debug information to be shown
        :param sequences: CSV data, to be partitioned for testing
        :return: list of dictionaries containing clf_name, predictions, time needed to predict
        """
        results = []
        predictions = []
        for ap in self.ap_list:
            ap_r, ap_p = ap.predict(sequences,  verbose)
            results = results + ap_r
            predictions = predictions + ap_p
        return results, predictions


    def explain(self, sequences: list, verbose:bool = True) -> list:
        """

        """
        explanations = []
        for ap in self.ap_list:
            ap_e = ap.explain(sequences,  verbose)
            explanations = explanations + ap_e
        return explanations