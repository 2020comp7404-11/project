from deepctr.models import xDeepFM
from deepctr.feature_column import SparseFeat, DenseFeat,get_feature_names
from sklearn.metrics import log_loss, roc_auc_score
from utils import *

if __name__ == "__main__":
  train, test, train_model_input, test_model_input, dnn_feature_columns, linear_feature_columns, feature_names, target = read_data_as_model()
  model = xDeepFM(linear_feature_columns,dnn_feature_columns,task='binary')
  model.compile("adam", "binary_crossentropy",
                metrics=['binary_crossentropy'], )

  history = model.fit(train_model_input, train[target].values,
                      batch_size=256, epochs=10, verbose=2, validation_split=0.2, )
  pred_ans = model.predict(test_model_input, batch_size=256)
  print("test LogLoss", round(log_loss(test[target].values, pred_ans), 4))
  print("test AUC", round(roc_auc_score(test[target].values, pred_ans), 4))
