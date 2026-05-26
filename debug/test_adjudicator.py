import os
import os
import random
import time

import numpy
import pandas
import sklearn
from confens.classifiers.ConfidenceBoosting import ConfidenceBoosting
from pandas import read_csv
from pyod.models.cblof import CBLOF
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.mcd import MCD
from pyod.models.pca import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier

# Sets random seed to increase repeatability
random.seed(23)
numpy.random.seed(23)
random_state = 42

# GLOBAL_VARS
LABEL_NAME = "Steer_Warning"
TRAIN_FILE = "output/scores_filtrato_wDecisiontraindata.csv"
TEST_FILE = "output/scores_filtrato_wDecisiontestdata.csv"
SCORES_FILE = "output/adj_scores_filtrato_wDecision.csv"

def current_ms() -> int:
    """
    Reports the current time in milliseconds
    :return: long int
    """
    return round(time.time() * 1000)

def read_dataset(path:str, label_name:str):
    """
    Reads dataset and separates it from label
    :param path:
    :param label_name:
    :return:
    """
    my_dataset = read_csv(path)
    label_obj = my_dataset[label_name].to_numpy()
    data_obj = my_dataset.drop(columns=[label_name])
    return data_obj.to_numpy(), label_obj, data_obj


if __name__ == "__main__":
    """
    Main of the data analysis
    """

    existing_exps = None
    if os.path.exists(SCORES_FILE):
        existing_exps = pandas.read_csv(SCORES_FILE)
        existing_exps = existing_exps.loc[:, ['dataset_tag', 'label_name', 'clf', 'contamination']]
    else:
        with open(SCORES_FILE, 'w') as f:
            f.write("dataset_tag,label_name,contamination,clf,tp,fp,fn,tn,mcc,acc,train_time\n")

    # load dataset PANDAS / NUMPY
    train_data, train_label, train_df = read_dataset(TRAIN_FILE, LABEL_NAME)
    test_data, test_label, test_df = read_dataset(TEST_FILE, LABEL_NAME)

    # Contamination
    outliers_fraction = numpy.average(train_label)
    if outliers_fraction > 0.5:
        outliers_fraction = 0.5
    print("Contamination is %.4f" % outliers_fraction)

    # choose classifier from PYOD, set of classifiers that I want to run and compare
    classifiers = {
        'Average KNN': KNN(method='mean',
                           contamination=outliers_fraction),
        'Isolation Forest': IForest(contamination=outliers_fraction,
                                    random_state=random_state),
        'Minimum Covariance Determinant (MCD)': MCD(
            contamination=outliers_fraction, random_state=random_state),
        'Principal Component Analysis (PCA)': PCA(
            contamination=outliers_fraction, random_state=random_state),
        'Cluster-based Local Outlier Factor (CBLOF)':
            CBLOF(contamination=outliers_fraction,
                  check_estimator=False, random_state=random_state),
        'DT': DecisionTreeClassifier(),
        'RF': RandomForestClassifier(),
        'ConfBoost': ConfidenceBoosting(n_base=20, clf=ExtraTreeClassifier()),
    }

    clf_index = 0
    for clf_name in classifiers.keys():
        clf_index += 1
        clf = classifiers[clf_name]

        if existing_exps is not None and (((existing_exps['dataset_tag'] == TRAIN_FILE) &
                                           (existing_exps['label_name'] == LABEL_NAME) &
                                           (existing_exps['contamination'] == outliers_fraction) &
                                           (existing_exps['clf'] == clf_name)).any()):
            print('%d/%d Skipping classifier %s, already in the results' % (clf_index, len(classifiers), clf_name))

        else:
            try:
                # Training an algorithm
                before_train = current_ms()
                clf.fit(train_data, train_label)
                after_train = current_ms()

                # Testing the trained model.
                predicted_labels = clf.predict(test_data)
                end = current_ms()

                # Computing metrics to understand how good an algorithm is
                accuracy = sklearn.metrics.accuracy_score(test_label, predicted_labels)
                mcc = sklearn.metrics.matthews_corrcoef(test_label, predicted_labels)
                tn, fp, fn, tp = confusion_matrix(test_label, predicted_labels).ravel()
                print("%d/%d %s: \t MCC is %.4f, Accuracy is %.4f, \ttrain time: %d, test time: %d \tTP: %d, TN: %d, FN: %d, FP: %d" % (
                    clf_index, len(classifiers), clf_name, mcc, accuracy, after_train - before_train, end - after_train, tp, tn, fn, fp))

                # Print scores file
                with open(SCORES_FILE, "a") as myfile:
                    myfile.write(TRAIN_FILE + "," + LABEL_NAME + "," + str(outliers_fraction) + "," + clf_name + "," +
                                 str(tp) + ',' + str(fp) + ',' + str(fn) + ',' + str(tn) + ',' +
                                 str(mcc) + "," + str(accuracy) + "," +
                                 str(after_train - before_train) + "," + str(end - after_train) + "\n")
            except:
                print("%d/%d %s: Error" % (clf_index, len(classifiers), clf_name))
