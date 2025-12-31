
import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import json
import os
from paths import PathRegistry

PATHS = PathRegistry.paths()

iris = load_iris()
X, y = iris.data, iris.target

# train model:
model = RandomForestClassifier(n_estimators=100, max_depth=3)
model.fit(X, y)

joblib.dump(model, os.path.join(PATHS['models'], 'iris_cls_1.joblib'))