import numpy
import shap
from sklearn.pipeline import Pipeline


class PredictionExplainer:

    def __init__(self,model):
        self.model = model
        self.explainer_obj = None

    def train(self, X, feature_names):
        pass

    def explain(self, X):
        pass

class SHAPExplainer(PredictionExplainer):
    """
    Disclaimer: on my laptop it requires 3-8ms per data point in test set depending on how complex the classifier is
    """

    def __init__(self, model):
        super().__init__(model)

    def train(self, X, feature_names):
        shap_model = self.model
        if isinstance(shap_model, Pipeline):
            shap_model = shap_model.steps[1][1]
        masker = shap.maskers.Independent(data=X)
        self.explainer_obj = shap.Explainer(self.model.predict_proba, masker)
        #self.explainer_obj = shap.Explainer(shap_model, X, feature_names=feature_names)

    def explain(self, X):
        exps = self.explainer_obj(X)
        arr_exp = numpy.asarray(exps.abs[:, :, 0])
        return arr_exp