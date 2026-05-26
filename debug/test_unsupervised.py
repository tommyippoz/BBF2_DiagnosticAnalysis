import copy
import os
import random
import time

import numpy
import pandas
import sklearn
from confens.classifiers.ConfidenceBagging import ConfidenceBagging
from confens.classifiers.ConfidenceBoosting import ConfidenceBoosting
from pandas import read_csv
from pyod.models.cblof import CBLOF
from pyod.models.copod import COPOD
from pyod.models.ecod import ECOD
from pyod.models.feature_bagging import FeatureBagging
from pyod.models.gmm import GMM
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.inne import INNE
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.mcd import MCD
from pyod.models.pca import PCA
from pyod.models.qmcd import QMCD
from pyod.models.sampling import Sampling
from sklearn.metrics import confusion_matrix

# Sets random seed to increase repeatability
random.seed(23)
numpy.random.seed(23)
random_state = 42

# GLOBAL_VARS
LABEL_NAME = "Steer_Error_1"
DATASET_NAME = "output_steererr_50"
SCORES_FILE = "output/scores_steererr_wDecision.csv"
PRINT_TEST = True

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
    my_dataset = my_dataset.sort_values('time')
    label_obj = my_dataset[label_name].to_numpy()
    data_obj = my_dataset.drop(columns=[label_name, "time", "Steer_Error_1", "Steer_Error_2",
                                        "name", "log_name", "addr", "fileID", "Steer_Warning"])
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
    train_data, train_label, train_df = read_dataset("./datasets/" + DATASET_NAME + "_train.csv", LABEL_NAME)
    test_data, test_label, test_df = read_dataset("./datasets/" + DATASET_NAME + "_test.csv", LABEL_NAME)
    train_df = copy.deepcopy(train_df)
    train_df[LABEL_NAME] = train_label
    test_df = copy.deepcopy(test_df)
    test_df[LABEL_NAME] = test_label

    # Contamination
    outliers_fraction = numpy.average(train_label)
    if outliers_fraction > 0.5:
        outliers_fraction = 0.5
    print("Contamination is %.4f" % outliers_fraction)

    # choose classifier from PYOD, set of classifiers that I want to run and compare
    classifiers = {

        #'One-class SVM (OCSVM)': OCSVM(contamination=outliers_fraction, kernel='linear'),
        #'Angle-based Outlier Detector (ABOD)':
        #    ABOD(contamination=outliers_fraction, n_neighbors=5, algorithm='ball_tree'),
        'K Nearest Neighbors (KNN)': KNN(
            contamination=outliers_fraction),
        'Average KNN': KNN(method='mean',
                           contamination=outliers_fraction),
        'Median KNN': KNN(method='median',
                          contamination=outliers_fraction),
        'Local Outlier Factor (LOF) 5':
            LOF(n_neighbors=5, contamination=outliers_fraction),
        'Local Outlier Factor (LOF) 15':
            LOF(n_neighbors=15, contamination=outliers_fraction),
        'Local Outlier Factor (LOF) 35':
            LOF(n_neighbors=35, contamination=outliers_fraction),

        'Isolation Forest': IForest(contamination=outliers_fraction,
                                    random_state=random_state),
        'INNE': INNE(
            max_samples=2, contamination=outliers_fraction,
            random_state=random_state,
        ),
        'Feature Bagging':
            FeatureBagging(LOF(n_neighbors=35),
                           contamination=outliers_fraction,
                           random_state=random_state),
        #'SUOD': SUOD(contamination=outliers_fraction),

        'Minimum Covariance Determinant (MCD)': MCD(
            contamination=outliers_fraction, random_state=random_state),

        'Principal Component Analysis (PCA)': PCA(
            contamination=outliers_fraction, random_state=random_state),

        'Probabilistic Mixture Modeling (GMM)': GMM(contamination=outliers_fraction,
                                                    random_state=random_state),

        'Histogram-based Outlier Detection (HBOS) 10': HBOS(
            contamination=outliers_fraction, n_bins=10),

        'Histogram-based Outlier Detection (HBOS) 20': HBOS(
            contamination=outliers_fraction, n_bins=20),

        'Copula-base Outlier Detection (COPOD)': COPOD(
            contamination=outliers_fraction),

        'ECDF-baseD Outlier Detection (ECOD)': ECOD(
            contamination=outliers_fraction),
        #'Kernel Density Functions (KDE)': KDE(contamination=outliers_fraction),

        'QMCD': QMCD(
            contamination=outliers_fraction),

        'Sampling': Sampling(
            contamination=outliers_fraction),

        'Cluster-based Local Outlier Factor (CBLOF)':
            CBLOF(contamination=outliers_fraction,
                  check_estimator=False, random_state=random_state),
    }
    base_algs = copy.deepcopy(list(classifiers.keys()))
    for clf_name in base_algs:
        classifiers["ConfBag10[" + clf_name + "]"] = (
            ConfidenceBagging(clf=classifiers[clf_name], n_base=10))
        classifiers["ConfBoost10[" + clf_name + "]"] = (
            ConfidenceBoosting(clf=classifiers[clf_name], n_base=10))

    classifiers['ConfBoost Mix 10'] = ConfidenceBoosting(
        clf=[LOF(n_neighbors=15, contamination=outliers_fraction),
             COPOD(contamination=outliers_fraction)], n_base=10)
    classifiers['ConfBoost Mix 20'] = ConfidenceBoosting(
        clf=[LOF(n_neighbors=15, contamination=outliers_fraction),
             COPOD(contamination=outliers_fraction)], n_base=20)
    classifiers['ConfBag Mix 10'] = ConfidenceBagging(
        clf=[LOF(n_neighbors=15, contamination=outliers_fraction),
             COPOD(contamination=outliers_fraction)], n_base=10)
    classifiers['ConfBag Mix 20'] = ConfidenceBagging(
        clf=[LOF(n_neighbors=15, contamination=outliers_fraction),
             COPOD(contamination=outliers_fraction)], n_base=20)

    clf_index = 0
    for clf_name in classifiers.keys():
        clf_index += 1
        clf = classifiers[clf_name]

        if existing_exps is not None and (((existing_exps['dataset_tag'] == DATASET_NAME) &
                                           (existing_exps['label_name'] == LABEL_NAME) &
                                           (existing_exps['contamination'] == outliers_fraction) &
                                           (existing_exps['clf'] == clf_name)).any()):
            print('%d/%d Skipping classifier %s, already in the results' % (clf_index, len(classifiers), clf_name))

        else:
            try:
                # Training an algorithm
                before_train = current_ms()
                clf.fit(train_data)
                after_train = current_ms()

                # Testing the trained model.
                predicted_labels = clf.predict(test_data)
                end = current_ms()

                # Building train files
                predicted_train = clf.predict(train_data)
                dec_funct = clf.decision_function(train_data)
                probas = clf.predict_proba(train_data)
                train_df["[PRED]" + clf_name] = predicted_train
                train_df["[SCORE]" + clf_name] = dec_funct
                train_df["[PROBA]" + clf_name] = probas[:, 1]

                # Building Test file
                dec_funct = clf.decision_function(test_data)
                probas = clf.predict_proba(test_data)
                test_df["[PRED]" + clf_name] = predicted_labels
                test_df["[SCORE]" + clf_name] = dec_funct
                test_df["[PROBA]" + clf_name] = probas[:, 1]

                # Computing metrics to understand how good an algorithm is
                accuracy = sklearn.metrics.accuracy_score(test_label, predicted_labels)
                mcc = sklearn.metrics.matthews_corrcoef(test_label, predicted_labels)
                tn, fp, fn, tp = confusion_matrix(test_label, predicted_labels).ravel()
                print("%d/%d %s: MCC is %.4f, Accuracy is %.4f, train time: %d, test time: %d TP: %d, TN: %d, FN: %d, FP: %d" % (
                    clf_index, len(classifiers), clf_name, mcc, accuracy, after_train - before_train, end - after_train, tp, tn, fn, fp))

                # Print scores file
                with open(SCORES_FILE, "a") as myfile:
                    myfile.write(DATASET_NAME + "," + LABEL_NAME + "," + str(outliers_fraction) + "," + clf_name + "," +
                                 str(tp) + ',' + str(fp) + ',' + str(fn) + ',' + str(tn) + ',' +
                                 str(mcc) + "," + str(accuracy) + "," +
                                 str(after_train - before_train) + "," + str(end - after_train) + "\n")
            except:
                print("%d/%d %s: Error" % (clf_index, len(classifiers), clf_name))

    if PRINT_TEST:
        train_df.to_csv(SCORES_FILE.replace(".csv", "traindata.csv"), index=False)
        test_df.to_csv(SCORES_FILE.replace(".csv", "testdata.csv"), index=False)