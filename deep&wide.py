# -*- coding: utf-8 -*-
"""
Created on Sat Nov 20 21:41:49 2020
@author: pengyifeng
Implement of Deep and Wide module
"""
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from deepctr.models.wdl import WDL
from deepctr.feature_column import SparseFeat, DenseFeat, get_feature_names
from utils import read_data_as_model

if __name__ == "__main__":
  train, test, train_model_input, test_model_input, dnn_feature_columns, linear_feature_columns, feature_names, target = read_data_as_model()
  model = WDL(linear_feature_columns, dnn_feature_columns, task='binary')
  model.compile("adam", "binary_crossentropy", 
                metrics=['binary_crossentropy'], )
  history = model.fit(train_model_input, train[target].values,
                      batch_size=256, epochs=10, verbose=2, validation_split=0.2, )
  pred_ans = model.predict(test_model_input, batch_size=256)  
