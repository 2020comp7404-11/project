import time
from logistic_example import run_logistic_model
from deep_wide import run_deep_wide_model
from deepFM import run_deepfm_model
from fm import run_fm_model
from xdeepfm import run_xdeepfm_model
from utils import plot_roc
from sklearn.metrics import roc_curve

if __name__ == '__main__':
    fpr_list, tpr_list, auc_list, name_list, timing_list = [], [], [], [], []
    func_list = [run_logistic_model, run_deep_wide_model, run_deepfm_model, run_xdeepfm_model, run_fm_model]
    for func in func_list:
        print('Running', func.__name__)
        t = time.process_time()
        pred_ans, y_test, auc, model_name = func()
        elapsed_time = time.process_time() - t
        timing_list.append((func.__name__, elapsed_time))
        fpr, tpr, thresholds = roc_curve(y_test, pred_ans, pos_label=0)
        fpr_list.append(fpr)
        tpr_list.append(tpr)
        auc_list.append(auc)
        name_list.append(model_name)
    [print(t[0], ': ', t[1], 's') for t in timing_list]
    plot_roc(tpr_list, fpr_list, auc_list, name_list)

