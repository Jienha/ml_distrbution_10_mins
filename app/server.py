from fastapi import FastAPI
import joblib
import os
import numpy as np

MODEL_SERIE = 'iris_cls_1.joblib'

model = joblib.load('models/{}'.format(MODEL_SERIE))

class_names = np.array(['setosa', 'versicolor', 'virginica'])

app = FastAPI()

@app.get('/')
def read_rood():
    return {'message': 'Iris model API'}

@app.post('/predict')
def predict(data: dict):
    features = np.array(data['features'].reshape(1, -1))
    prediction = model.predict(features)
    class_name = class_names[prediction][0]
    return {'predicted_class': class_name}
