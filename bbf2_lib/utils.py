import configparser
import math
import os
import time

import numpy
import sklearn.metrics
from sklearn.metrics import confusion_matrix

from bbf2_lib.AnomalyPredictor import AnomalyPredictor, TimeSeriesAnomalyPredictor


# ----------------------------------------------------
#          FILE CONTAINING UTILITY FUNCTIONS
# ----------------------------------------------------


def read_config(cfg_file):
    """
    Loads parameters for execution from a configuration file.
    :param cfg_file: the path to the configuration file
    :return: a dictionary
    """
    # Loading Conf file
    if not os.path.exists(cfg_file):
        return None
    config = configparser.ConfigParser()
    config.read(cfg_file)

    # Setting up variables
    cfg_info = {}
    for section in config.sections():
        cfg_info[section] = {}
        for option in config.options(section):
            cfg_info[option] = config.get(section, option)
            if "," in cfg_info[option]:
                cfg_info[option] = [x.strip() for x in cfg_info[option].split(',')]
            elif cfg_info[option] in ["True", "False", "true", "false", "t", "f", "T", "F"]:
                cfg_info[option] = True if cfg_info[option].upper().startswith("T") else False

    return cfg_info


def current_ms():
    """
    Reports the current time in milliseconds
    :return: long int
    """
    return round(time.time() * 1000)


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
    stats['point']['tn'], stats['point']['fp'], stats['point']['fn'], stats['point']['tp'] = (
        confusion_matrix(test_y, pred_y).ravel())
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

def test_models(predictor: AnomalyPredictor, test_sequences:list, verbose:bool = True) -> list:
    """
    Tests a group of models trained with an AnomalyPredictor
    :param predictor: the anomalypredictor of choice (has to be fitted already)
    :param test_sequences: test sequences
    :param verbose: True if debug information has to be shown
    :return: a list of dict containing metrics
    """
    y_test = numpy.concatenate([numpy.asarray(item["Y"], dtype=int) for item in test_sequences], axis=0)
    test_results = predictor.predict(sequences=test_sequences)
    if verbose:
        print("----- PREDICTION RESULTS -----")
    metrics_list = []
    for result in test_results:
        metrics = score_metrics(result["predictions"], y_test, test_sequences)
        metrics["general"] = {"clf": result["clf"], "predict_time": result["predict_time"],
                              "predict_time_per_item": result["predict_time_per_item"],
                              "use_timeseries":isinstance(predictor, TimeSeriesAnomalyPredictor),
                              "supervised": predictor.supervised}
        metrics_list.append(metrics)
        if verbose:
            print("Classifier %s predicted with accuracy %.4f" % (result["clf"], metrics["point"]["accuracy"]))

    return metrics_list


def print_scores(to_print, analysis_tag:str, output_folder, filename):
    """
    Prints scores of test runs to file specified in the CFG
    :param to_print: data to print
    :param output_folder: folder in which outputs are placed
    :param filename: name of the output file
    :return:
    """
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    csv_file = os.path.join(output_folder, filename).replace('"', '')
    if not os.path.exists(csv_file):
        with open(csv_file, 'w') as f:
            f.write("timeseries,supervised,analysis_tag,clf_name,tp,fp,fn,tn,mcc,acc,recall,test_time,test_time_per_item,dd_min,dd_max,dd_avg,fpr_min,fpr_max,fpr_avg,tpr_min,tpr_max,tpr_avg\n")
    for result in to_print:
        with open(csv_file, "a") as myfile:
            myfile.write(str(result['general']['use_timeseries']) + "," +
                         str(result['general']['supervised']) + "," +
                         analysis_tag + "," + result['general']['clf'] + "," +
                         str(result['point']['tp']) + "," + str(result['point']['fp']) + "," +
                         str(result['point']['fn']) + "," + str(result['point']['tn']) + "," +
                         str(result['point']['mcc']) + "," + str(result['point']['accuracy']) + "," +
                         str(result['point']['recall']) + ',' + str(result['general']['predict_time']) + ',' +
                         str(result['general']['predict_time_per_item']) + "," +
                         str(result['timeseries']['dd']['min']) + "," +
                         str(result['timeseries']['dd']['max']) + "," +
                         str(result['timeseries']['dd']['avg']) + "," +
                         str(result['timeseries']['fpr']['min']) + "," +
                         str(result['timeseries']['fpr']['max']) + "," +
                         str(result['timeseries']['fpr']['avg']) + "," +
                         str(result['timeseries']['tpr']['min']) + "," +
                         str(result['timeseries']['tpr']['max']) + "," +
                         str(result['timeseries']['tpr']['avg']) + "\n")

    pass