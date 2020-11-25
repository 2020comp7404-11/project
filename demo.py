from logistic_example import run_logistic_model
from deep_wide import run_deep_wide_model
from deepFM import run_deepfm_model
from xdeepfm import run_xdeepfm_model
from utils import plot_roc
from sklearn.metrics import roc_curve

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

