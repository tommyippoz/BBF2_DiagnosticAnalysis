import numpy
import shap
from sklearn.pipeline import Pipeline


class PredictionExplainer:
    """
    Abstract class that serves as a meta-model for prediction explainers / diagnosis
    """

    def __init__(self,model):
        """
        Constructor of abstract class
        :param model: the model to explain
        """
        self.model = model
        self.explainer_obj = None

    def train(self, X, feature_names):
        """
        Trains the explainer
        :param X: train set
        :param feature_names: names of dataset features
        :return:
        """
        pass

    def explain(self, X) -> numpy.ndarray:
        """
        Explains the predictions contained in X, returns a numpy array with numeric explanations for each data point and feature
        :param X: test set
        :return:
        """
        pass

class SHAPExplainer(PredictionExplainer):
    """
    Disclaimer: on my laptop it requires 3-8ms per data point in test set depending on how complex the classifier is
    """

    def __init__(self, model):
        super().__init__(model)

    def train(self, X, feature_names):
        """
        Trains the explainer
        :param X: train set
        :param feature_names: names of dataset features
        :return:
        """
        masker = shap.maskers.Independent(data=X)
        self.explainer_obj = shap.Explainer(self.model.predict_proba, masker)

    def explain(self, X) -> numpy.ndarray:
        """
        Explains the predictions contained in X, returns a numpy array with numeric explanations for each data point and feature
        :param X: test set
        :return:
        """
        exps = self.explainer_obj(X)
        arr_exp = numpy.asarray(exps.abs[:, :, 0].values)
        # Normalize
        sums = arr_exp.sum(axis=1, keepdims=True)
        arr_exp = numpy.asarray([arr_exp[i, :] / sums[i] for i in range(0, len(sums))])
        return arr_exp