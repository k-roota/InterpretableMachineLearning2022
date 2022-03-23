from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from tqdm import tqdm
import xgboost
import warnings
import numpy as np
import pandas as pd
import dalex as dx


if __name__ == '__main__':

    warnings.filterwarnings('ignore')
    np.random.seed(1)

    data = load_breast_cancer(as_frame=True)
    data, target = data["data"], data["target"]
    data = (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0))

    data_train, data_test, target_train, target_test = train_test_split(data, target, test_size=0.1)

    # train xgboost and calculate prediction
    xgboost_model = xgboost.XGBClassifier(use_label_encoder=False, eval_metric="logloss").fit(data_train, target_train)
    results = pd.DataFrame({
        "prediction": xgboost_model.predict(data_test),
        "target": target_test
    })
    results["match"] = results["prediction"] == results["target"]

    # choose interesting samples (41 - M predicted as B, 29 - correct M, 285 - correct B)
    interesting_predictions = [29, 41, 285]
    print(results.loc[interesting_predictions])

    # generate SHAP explanation for 3 predictions
    titles = [
        "Example of malignant tumor predicted correctly by XGBoost [patient 29].",
        "Example of malignant tumor predicted as benign by XGBoost [patient 41].",
        "Example of benign tumor predicted correctly by XGBoost [patient 285]."
    ]
    xgboost_explainer = dx.Explainer(xgboost_model, data_test, target_test)
    for i, prediction_no in enumerate(interesting_predictions):
        explanation = xgboost_explainer.predict_parts(data_test.loc[[prediction_no]], type="shap")
        fig = explanation.plot(show=False, title=titles[i])
        fig.write_image(f"resources/shap_prediction{prediction_no}.png")

    # generate SHAP explanation with different "highest importance" features (auto choice)
    interesting_predictions = [47, 421]
    print(results.loc[interesting_predictions])

    titles = [
        "Example of malignant tumor predicted correctly by XGBoost [patient 47].",
        "Example of correct benign tumor prediction by XGBoost [patient 421]."
    ]
    for i, prediction_no in enumerate(interesting_predictions):
        explanation = xgboost_explainer.predict_parts(data_test.loc[[prediction_no]], type="shap")
        fig = explanation.plot(show=False, title=titles[i])
        fig.write_image(f"resources/various_importance{prediction_no}.png")

    # generate SHAP explanation where same feature has +/- (omitted due to worst area 29 and 41)
    # train MLP and calculate predictions
    mlp_model = MLPClassifier(max_iter=1000).fit(data_train, target_train)
    mlp_explainer = dx.Explainer(mlp_model, data_test, target_test)
    explanation = mlp_explainer.predict_parts(data_test.loc[[41]], type="shap")
    fig = explanation.plot(show=False, title="Example of malignant tumor predicted correctly by MLP [patient 41].")
    fig.write_image(f"resources/shap_mlp_prediction41.png")
