# -*- coding: utf-8 -*-
"""
Created on Sat Nov 20 21:41:49 2020
@author: pengyifeng
Implement of Deep and Wide module
"""
from sklearn.metrics import log_loss, roc_auc_score
from deepctr.models.wdl import WDL
from utils import read_data_as_model



def run_deep_wide_model():
  train, test, train_model_input, test_model_input, dnn_feature_columns, linear_feature_columns, feature_names, target = read_data_as_model()
  model = WDL(linear_feature_columns, dnn_feature_columns, task='binary')
  model.compile("adam", "binary_crossentropy",
                metrics=['binary_crossentropy'], )
  model.fit(train_model_input, train[target].values,
                      batch_size=256, epochs=10, verbose=2, validation_split=0.2, )
  pred_ans = model.predict(test_model_input, batch_size=256)
  print("test LogLoss", round(log_loss(test[target].values, pred_ans), 4))
  print("test AUC", round(roc_auc_score(test[target].values, pred_ans), 4))
  return pred_ans, test[target].values, round(roc_auc_score(test[target].values, pred_ans), 4), 'wide&deep'

if __name__ == "__main__":
  run_deep_wide_model()
