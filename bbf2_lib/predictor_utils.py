import json
import math
import os

import joblib
import numpy
import sklearn
from sklearn.metrics import confusion_matrix

from bbf2_lib.AnomalyPredictor import TimeSeriesAnomalyPredictor, AnomalyPredictor
from bbf2_lib.Classifier import get_classifier_name
from bbf2_lib.utils import clear_folder


def score_metrics(pred_y, test_y, test_sequences=None,
                         diagnosis_time: int = 1, max_delay: int = 50) -> dict:
    """
    Computes stats for predicted labels
    :param pred_y: scores of the classifier
    :param test_y: ground truth
    :param classes: problem's classes
    :return: a dictionary
    """
    stats = {"point":{}, "timeseries":{}}
    c_mat = confusion_matrix(test_y, pred_y)
    stats['point']['tn'] = stats['point']['fp'] = stats['point']['fn'] = stats['point']['tp'] = 0
    stats['point']['tn'] = c_mat[0]
    if len(c_mat) > 1:
        stats['point']['fp'] = c_mat[1]
        if len(c_mat) == 4:
            stats['point']['fn'] = c_mat[2]
            stats['point']['tp'] = c_mat[3]
    stats['point']['accuracy'] = sklearn.metrics.balanced_accuracy_score(test_y, pred_y)
    stats['point']['mcc'] = sklearn.metrics.matthews_corrcoef(test_y, pred_y)
    stats['point']['recall'] = sklearn.metrics.recall_score(test_y, pred_y)
    if test_sequences is not None:
        normal_class = 0
        fprs = []
        dds = []
        tprs = []
        index = 0
        for seq in test_sequences:

            seq_data = seq["X"]
            seq_labels = seq["Y"]
            # Deriving sequence-related scores
            seq_pred = pred_y[index:index + len(seq_labels)]
            seq_label = test_y[index:index + len(seq_labels)]
            index += len(seq_labels)

            # Iterating over sequence data
            cooldown = 0
            an_time = -1
            det_time = -1
            false_alerts = []
            for i in range(0, len(seq_pred)):
                if cooldown == 0 and seq_pred[i] != normal_class:
                    cooldown = diagnosis_time
                    if det_time < 0 and seq_label[i] == seq_pred[i]:
                        det_time = i
                    if seq_label[i] != seq_pred[i]:
                        false_alerts.append(i)
                if an_time < 0 and seq_label[i] != normal_class:
                    an_time = i
                if cooldown > 0:
                    cooldown -= 1

            # Updating metrics
            fprs.append(len(false_alerts) / (seq_label == normal_class).sum())
            detected = 1 if det_time > 0 and not math.isnan(det_time) else 0
            tprs.append(detected)
            if detected > 0:
                dds.append((det_time - an_time) if an_time <= det_time and not math.isnan(det_time) else max_delay)

        stats['timeseries']['fpr'] = {'avg': float(numpy.average(fprs)), 'min': float(numpy.min(fprs)),
                                'max': float(numpy.max(fprs)), 'median': float(numpy.median(fprs)),  # 'all': fprs
                                }
        stats['timeseries']['tpr'] = {'avg': float(numpy.average(tprs)), 'min': int(numpy.min(tprs)),
                                'max': int(numpy.max(tprs)), 'median': int(numpy.median(tprs)),  # 'all': tprs
                                }
        stats['timeseries']['dd'] = {'avg': float(numpy.nanmean(dds)), 'min': int(numpy.nanmin(dds)),
                               'max': int(numpy.nanmax(dds)), 'median': int(numpy.nanmedian(dds)),  # 'all': dds
                               } if len(dds) > 0 else {'avg': -1, 'min': -1, 'max': -1, 'median': -1}

    return stats

# --------------------------- UTILITY METHODS ------------------------------------------

def test_models(predictor: AnomalyPredictor, test_sequences:list, verbose:bool = True) -> list:
    """
    Tests a group of models trained with an AnomalyPredictor
    :param predictor: the anomalypredictor of choice (has to be fitted already)
    :param test_sequences: test sequences
    :param verbose: True if debug information has to be shown
    :return: a list of dict containing metrics
    """
    y_test = numpy.concatenate([numpy.asarray(item["Y"], dtype=int) for item in test_sequences], axis=0)
    test_results, predictions = predictor.predict(sequences=test_sequences)
    if verbose:
        print("----- PREDICTION RESULTS -----")
    metrics_list = []
    for result in test_results:
        metrics = score_metrics(result["predictions"], y_test, test_sequences)
        metrics["general"] = {"clf": result["clf"],
                              "model": result["model"],
                              "predict_time": result["predict_time"],
                              "predict_time_per_item": result["predict_time_per_item"],
                              "use_timeseries":isinstance(predictor, TimeSeriesAnomalyPredictor),
                              "supervised": predictor.supervised}
        metrics_list.append(metrics)
        if verbose:
            print("Classifier %s predicted with accuracy %.4f" % (result["clf"], metrics["point"]["accuracy"]))

    return metrics_list, predictions

def store_models(clf_scores: list, models_folder: str = "./models", use_timeseries: bool = True):
    """
    Stores classifier models, each in a specific folder
    :param clf_scores: the scores on test set
    :param models_folder: the folder where the models are stored
    :return:
    """
    if not os.path.exists(models_folder):
        os.mkdir(models_folder)
    models_folder = os.path.join(models_folder, "timeseries" if use_timeseries else "point")
    if not os.path.exists(models_folder):
        os.mkdir(models_folder)
    for clf_data in clf_scores:
        clf = clf_data["general"]["model"]
        models_details_folder = os.path.join(models_folder, get_classifier_name(clf))
        if not os.path.exists(models_details_folder):
            os.mkdir(models_details_folder)
        else:
            clear_folder(models_details_folder)

        model_file = os.path.join(models_details_folder, "model.joblib")
        joblib.dump(clf, model_file, compress=9)

        # Tests if storing was successful
        clf_obj = joblib.load(model_file)
        if get_classifier_name(clf) == get_classifier_name(clf_obj):
            print(" %s Model stored successfully at '%s'" % (str(clf_data["general"]["clf"]), model_file))
            clf_data["general"].pop('model', None)
            json_metrics = json.dumps(clf_data, cls=NpEncoder)
            with open(model_file.replace("model.joblib", "model_stats.json"), "w") as f:
                f.write(json_metrics)
        else:
            print("Error while storing the model - file corrupted")
        pass

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.integer):
            return int(obj)
        if isinstance(obj, numpy.floating):
            return float(obj)
        if isinstance(obj, numpy.ndarray):
            return obj.tolist()
        if isinstance(obj, numpy.bool):
            return obj == True
        return super(NpEncoder, self).default(obj)