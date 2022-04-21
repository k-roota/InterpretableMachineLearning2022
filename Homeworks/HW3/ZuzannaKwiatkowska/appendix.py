from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import xgboost
import warnings
import numpy as np
import pandas as pd
import lime.lime_tabular
import pickle

NUM_FEATURES = 5

if __name__ == '__main__':

    warnings.filterwarnings('ignore')
    np.random.seed(1)

    data = load_breast_cancer(as_frame=True)
    data, target = data["data"], data["target"]
    data = (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0))

    data_train, data_test, target_train, target_test = train_test_split(data, target, test_size=0.1)

    # train xgboost and calculate prediction
    xgboost_model = xgboost.XGBClassifier(use_label_encoder=False, eval_metric="logloss").fit(data_train, target_train)
    results_xgboost = pd.DataFrame({
        "prediction": xgboost_model.predict(data_test),
        "target": target_test
    })
    results_xgboost["match"] = results_xgboost["prediction"] == results_xgboost["target"]

    # choose interesting samples (misclassified, correct benign, correct malignant)
    interesting_predictions = pd.concat([
               results_xgboost[-results_xgboost["match"]].sample(n=1),
               results_xgboost[results_xgboost["match"] & (results_xgboost["target"] == 1)].sample(n=1),
               results_xgboost[results_xgboost["match"] & (results_xgboost["target"] == 0)].sample(n=1),
    ])

    # lime for couple of predictions (stability of decomposition)
    explainer = lime.lime_tabular.LimeTabularExplainer(
        data_test.values,
        feature_names=data_test.columns,
        class_names=["malignant", "benign"],
        discretize_continuous=True,
    )
    for ix in interesting_predictions.index:
        exp = explainer.explain_instance(data_test.loc[ix], xgboost_model.predict_proba, num_features=NUM_FEATURES)
        with open(f"resources/lime_xgboost_{ix}.pkl", "wb") as stream:
            pickle.dump(exp, stream)

    # train MLP and calculate predictions
    mlp_model = MLPClassifier(max_iter=1000).fit(data_train, target_train)

    # find an observation for which lime decomposition is different for 2 models
    for ix in interesting_predictions.index:
        exp = explainer.explain_instance(data_test.loc[ix].values, mlp_model.predict_proba, num_features=NUM_FEATURES)
        with open(f"resources/lime_mlp_{ix}.pkl", "wb") as stream:
            pickle.dump(exp, stream)
