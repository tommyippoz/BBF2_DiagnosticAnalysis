import numpy
import pandas
import pandas as pd
import pyod
import sklearn
from pyod.models.base import BaseDetector
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.utils import check_X_y
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_is_fitted, check_array
from xgboost import XGBClassifier


from confens.classifiers.ConfidenceBoosting import ConfidenceBoosting
from pyod.models.abod import ABOD
from pyod.models.cblof import CBLOF
from pyod.models.cof import COF
from pyod.models.copod import COPOD
from pyod.models.ecod import ECOD
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.inne import INNE
from pyod.models.knn import KNN
from pyod.models.loda import LODA
from pyod.models.lof import LOF
from pyod.models.mcd import MCD
from pyod.models.ocsvm import OCSVM
from pyod.models.pca import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from xgboost import XGBClassifier


# ---------------------------------- SUPPORT METHODS ------------------------------------

def get_classifier_name(clf):
    if isinstance(clf, Classifier):
        return clf.classifier_name()
    else:
        return clf.__class__.__name__


def get_feature_importance(clf):
    if isinstance(clf, Classifier):
        return clf.feature_importances()
    else:
        return clf.feature_importances_



def choose_classifiers(clf_names, contamination=None, verbose=False) -> list:
    """
    Returns classifiers object from string descriptions as list
    :param clf_names: list of string descriptions of classifiers
    :param contamination: used for unsupervised classifiers, fraction of anomalies in train set
    :param verbose: true if debug information has to be shown
    :return: a classifier object
    """
    clfs = []
    if not isinstance(clf_names, list):
        clf_names = [clf_names]
    for clf in clf_names:
        clf_obj = choose_classifier(clf, contamination)
        if clf_obj is not None:
            clfs.append(clf_obj)
            if verbose:
                print("\tAdded classifier: %s" % get_classifier_name(clf_obj))
    return clfs


def choose_classifier(clf_name, contamination=None):
    """
    Returns classifier object from string description
    :param clf_name: string description of the classifier
    :param contamination: used for unsupervised classifiers, fration of anomalies in train set
    :return: a classifier object
    """
    if contamination is not None and contamination > 0.5:
        contamination = 0.5
    if clf_name in {"XGB", "XGBoost"}:
        return XGBClassifier(use_label_encoder=False, eval_metric="logloss")
    elif clf_name in {"DT", "DTree", "DecisionTree"}:
        return DecisionTreeClassifier()
    elif clf_name in {"KNN", "knn", "kNN", "KNeighbours"}:
        return KNeighborsClassifier(n_neighbors=11, n_jobs=-1, algorithm="kd_tree")
    elif clf_name in {"SVM"}:
        return BaggingClassifier(SVC(gamma='auto', probability=True), max_samples=0.1, n_estimators=10)
    elif clf_name in {"LDA"}:
        return LinearDiscriminantAnalysis()
    elif clf_name in {"GNB", "GaussianNB"}:
        return Pipeline([("norm", MinMaxScaler()), ("clf", GaussianNB())])
    elif clf_name in {"MNB", "MultinomialNB"}:
        return Pipeline([("norm", MinMaxScaler()), ("clf", MultinomialNB())])
    elif clf_name in {"Regression", "LogisticRegression", "LR"}:
        return LogisticReg()
    elif clf_name in {"RF", "RandomForest"}:
        return RandomForestClassifier(n_estimators=10)
    elif clf_name in {"GBC", "GradientBoosting"}:
        return GradientBoostingClassifier(n_estimators=50)
    elif clf_name in {"CONFBOOST", "CBOOST", "CBoost"}:
        return ConfidenceBoosting(n_base=20, clf=ExtraTreeClassifier())
    elif clf_name in {"COPOD"}:
        return UnsupervisedClassifier(COPOD(contamination=contamination))
    elif clf_name in {"ECOD"}:
        return UnsupervisedClassifier(ECOD(contamination=contamination))
    elif clf_name in {"HBOS"}:
        return UnsupervisedClassifier(HBOS(contamination=contamination, n_bins=30))
    elif clf_name in {"MCD"}:
        return UnsupervisedClassifier(MCD(contamination=contamination, support_fraction=0.9))
    elif clf_name in {"PCA"}:
        return UnsupervisedClassifier(PCA(contamination=contamination))
    elif clf_name in {"CBLOF"}:
        return UnsupervisedClassifier(CBLOF(contamination=contamination, alpha=0.75, beta=3))
    elif clf_name in {"OCSVM", "1SVM"}:
        return UnsupervisedClassifier(OCSVM(contamination=contamination))
    elif clf_name in {"uKNN"}:
        return UnsupervisedClassifier(KNN(contamination=contamination, n_neighbors=3))
    elif clf_name in {"LOF"}:
        return UnsupervisedClassifier(LOF(contamination=contamination, n_neighbors=5))
    elif clf_name in {"INNE"}:
        return UnsupervisedClassifier(INNE(contamination=contamination))
    elif clf_name in {"FastABOD", "ABOD", "FABOD"}:
        return UnsupervisedClassifier(ABOD(contamination=contamination, method='fast', n_neighbors=7))
    elif clf_name in {"COF"}:
        return UnsupervisedClassifier(COF(contamination=contamination, n_neighbors=9))
    elif clf_name in {"IFOREST", "IForest"}:
        return UnsupervisedClassifier(IForest(contamination=contamination, n_estimators=10))
    elif clf_name in {"LODA"}:
        return UnsupervisedClassifier(LODA(contamination=contamination, n_bins=100))
    else:
        return None


class Classifier(BaseEstimator, ClassifierMixin):
    """
    Basic Abstract Class for Classifiers.
    Abstract methods are only the classifier_name, with many degrees of freedom in implementing them.
    Wraps implementations from different frameworks (if needed), sklearn and many deep learning utilities
    """

    def __init__(self, clf):
        """
        Constructor of a generic Classifier
        :param clf: algorithm to be used as Classifier
        """
        self.clf = clf
        self._estimator_type = "classifier"
        self.feature_importances_ = None
        self.X_ = None
        self.y_ = None

    def fit(self, X, y=None):

        # Check that X and y have correct shape
        X, y = check_X_y(X, y)

        # Store the classes seen during fit + other data
        if y is not None:
            self.classes_ = unique_labels(y)
        else:
            self.classes_ = [0, 1]
        #self.X_ = X
        #self.y_ = y

        # Train clf
        self.clf.fit(X, y)
        self.feature_importances_ = self.compute_feature_importances()

        # Return the classifier
        return self

    def feature_importances(self):
        """
        Returns feature importances
        :return: a list
        """
        return self.feature_importances_

    def predict(self, X):
        """
        Method to compute predict of a classifier
        :return: array of predicted class
        """
        probas = self.predict_proba(X)
        return self.classes_[numpy.argmax(probas, axis=1)]

    def predict_proba(self, X):
        """
        Method to compute probabilities of predicted classes
        :return: array of probabilities for each classes
        """

        # Check if fit has been called
        check_is_fitted(self)
        X = check_array(X)

        return self.clf.predict_proba(X)

    def predict_confidence(self, X):
        """
        Method to compute confidence in the predicted class
        :return: max probability as default
        """
        probas = self.predict_proba(X)
        return numpy.max(probas, axis=1)

    def compute_feature_importances(self):
        """
        Outputs feature ranking in building a Classifier
        :return: ndarray containing feature ranks
        """
        if hasattr(self.clf, 'feature_importances_'):
            return self.clf.feature_importances_
        elif hasattr(self.clf, 'coef_'):
            return numpy.sum(numpy.absolute(self.clf.coef_), axis=0)
        return []

    def classifier_name(self):
        """
        Returns the name of the classifier (as string)
        """
        return self.clf.__class__.__name__

    def get_params(self, deep=True):
        return {'clf': self.clf}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self.clf, parameter, value)
        return self

class UnsupervisedClassifier(Classifier, BaseDetector):
    """
    Wrapper for unsupervised classifiers belonging to the library PYOD
    """

    def __init__(self, clf):
        """
        Constructor of a generic UnsupervisedClassifier. Assumes that clf is an algorithm from pyod
        :param clf: pyod algorithm to be used as Classifier
        """
        self.clf = clf
        self.contamination = clf.contamination
        self._estimator_type = "classifier"
        self.feature_importances_ = None
        self.X_ = None
        self.y_ = None

    def fit(self, X, y=None):

        # Store the classes seen during fit + other data
        self.classes_ = [0, 1]
        self.X_ = X
        self.y_ = None

        # Train clf
        self.clf.fit(X)
        self.feature_importances_ = self.compute_feature_importances()

        # Return the classifier
        return self

    def decision_function(self, X):
        """
        pyod function to override. Calls the wrapped classifier.
        :param X: test set
        :return: decision function
        """
        return self.clf.decision_function(X)

    def predict_proba(self, X):
        """
        Method to compute probabilities of predicted classes.
        It has to e overridden since PYOD's implementation of predict_proba is wrong
        :return: array of probabilities for each classes
        """

        # Check if fit has been called
        check_is_fitted(self)
        X = check_array(X)

        pred_score = self.decision_function(X)
        probs = numpy.zeros((X.shape[0], 2))
        if isinstance(self.contamination, (float, int)):
            pred_thr = pred_score - self.clf.threshold_
        else:
            pred_thr = 0.5
        min_pt = min(pred_thr)
        max_pt = max(pred_thr)
        anomaly = pred_thr > 0
        cont = numpy.asarray([pred_thr[i] / max_pt if anomaly[i] else (pred_thr[i] / min_pt if min_pt != 0 else 0.2)
                              for i in range(0, len(pred_thr))])
        probs[:, 0] = 0.5 + cont / 2
        probs[:, 1] = 1 - probs[:, 0]
        probs[anomaly, 0], probs[anomaly, 1] = probs[anomaly, 1], probs[anomaly, 0]
        return probs

    def predict(self, X):
        """
        Method to compute predict of a classifier
        :return: array of predicted class
        """
        probas = self.predict_proba(X)
        return numpy.argmax(probas, axis=1)

    def classifier_name(self):
        """
        Returns the name of the classifier (as string)
        """
        return self.clf.__class__.__name__

class LogisticReg(Classifier):
    """
    Wrapper for the sklearn.LogisticRegression algorithm
    """

    def __init__(self):
        Classifier.__init__(self, LogisticRegression(solver='sag',
                                                     random_state=0,
                                                     multi_class='ovr',
                                                     max_iter=10000,
                                                     n_jobs=10,
                                                     tol=0.1))

    def classifier_name(self):
        return "LogisticRegression"

class XGB(Classifier):
    """
    Wrapper for the sklearn.LogisticRegression algorithm
    """

    def __init__(self, n_estimators=100):
        Classifier.__init__(self, XGBClassifier(n_estimators=n_estimators))
        self.l_encoder = None

    def fit(self, X, y=None):

        # Check that X and y have correct shape
        X, y = check_X_y(X, y)

        # Store the classes seen during fit + other data
        self.classes_ = unique_labels(y)
        self.l_encoder = LabelEncoder()
        y = self.l_encoder.fit_transform(y)

        #self.X_ = X
        #self.y_ = y

        # Train clf
        self.clf.fit(X, y)
        self.feature_importances_ = self.compute_feature_importances()

        # Return the classifier
        return self

    def classifier_name(self):
        return "XGBClassifier"
