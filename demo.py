from logistic_example import *
from deepFM import *
from xdeepfm import *
from deepctr.models.wdl import WDL
from utils import read_data_as_model

def run_logistic_model():
    x_train, x_test, y_train, y_test = read_data()
    logit_reg = LogisticRegression(C=1.0, fit_intercept=True, penalty='l2', dual=False,
                                   tol=1e-5, max_iter=5000)
    model = logistic_regression(model=logit_reg)
    model.fit(x_train, y_train)
    pred_ans = model.predict(x_test)

    test_logloss = round(log_loss(y_test, pred_ans), 4)
    test_auc = round(roc_auc_score(y_test, pred_ans), 4)
    print("test LogLoss", test_logloss, 4)
    print("test AUC", test_auc, 4)
    return pred_ans, y_test, test_auc, 'logistic'

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

def run_deepfm_model():
    train, test, train_model_input, test_model_input, dnn_feature_columns, linear_feature_columns, feature_names, target = read_data_as_model()

    #Define Model,train,predict and evaluate
    model = DeepFM(linear_feature_columns, dnn_feature_columns, task='binary')
    model.compile("adam", "binary_crossentropy",
                  metrics=['binary_crossentropy'], )

    model.fit(train_model_input, train[target].values,
                        batch_size=256, epochs=10, verbose=2, validation_split=0.2, )
    pred_ans = model.predict(test_model_input, batch_size=256)
    print("test LogLoss", round(log_loss(test[target].values, pred_ans), 4))
    print("test AUC", round(roc_auc_score(test[target].values, pred_ans), 4))
    return pred_ans, test[target].values, round(roc_auc_score(test[target].values, pred_ans), 4), 'deepfm'

def run_xdeepfm_model():
    train, test, train_model_input, test_model_input, dnn_feature_columns, linear_feature_columns, feature_names, target = read_data_as_model()
    model = xDeepFM(linear_feature_columns, dnn_feature_columns, task='binary')
    model.compile("adam", "binary_crossentropy",
                  metrics=['binary_crossentropy'], )

    model.fit(train_model_input, train[target].values,
                        batch_size=256, epochs=10, verbose=2, validation_split=0.2, )
    pred_ans = model.predict(test_model_input, batch_size=256)
    print("test LogLoss", round(log_loss(test[target].values, pred_ans), 4))
    print("test AUC", round(roc_auc_score(test[target].values, pred_ans), 4))
    return pred_ans, test[target].values, round(roc_auc_score(test[target].values, pred_ans), 4), 'xdeepfm'

if __name__ == '__main__':
    fpr_list, tpr_list, auc_list, name_list = [], [], [], []
    func_list = [run_logistic_model, run_deep_wide_model, run_deepfm_model, run_xdeepfm_model]
    for func in func_list:
        pred_ans, y_test, auc, model_name = func()
        fpr, tpr, thresholds = roc_curve(y_test, pred_ans, pos_label=0)
        fpr_list.append(fpr)
        tpr_list.append(tpr)
        auc_list.append(auc)
        name_list.append(model_name)
    plot_roc(tpr_list, fpr_list, auc_list, name_list)

