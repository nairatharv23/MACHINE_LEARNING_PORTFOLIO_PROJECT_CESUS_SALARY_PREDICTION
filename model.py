import pickle
import pandas as pd
import numpy as np
import os

curr_path = os.path.dirname(os.path.realpath(__file__))
model_path = os.path.join(curr_path,  'modelrandom.pkl')
model = pickle.load(open(model_path, 'rb'))

def predict_salary(attributes: np.ndarray):
    """ Returns 1 if customer salary is then than 50K, 0 otherwise"""
    # print(attributes.shape) # (1,10)
    




    pred = model.predict(attributes.reshape(1, -1))
    

    return int(pred[0])
