from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from binary_model import binary_model
from utils import *

class logistic_regression(binary_model):
    def __init__(self, model):
        super().__init__(model)

    def fit(self, x_train, y_train):
        self.model.fit(x_train, y_train)

    def predict(self, x_test):
        y_preds = self.model.predict_proba(x_test)[:,1]
        return y_preds

    def score(self, y_test, y_preds, pos_label=1):
        fpr, tpr, thresholds = metrics.roc_curve(y_test, y_preds, pos_label=pos_label)
        auc = metrics.auc(fpr, tpr)
        return auc

if __name__ == "__main__":
    x_train, x_test, y_train, y_test = read_data()
    logit_reg = LogisticRegression(C=1.0, fit_intercept=True, penalty='l2', dual=False,
                                   tol=1e-5, max_iter=5000)
    model = logistic_regression(model=logit_reg)
    model.fit(x_train, y_train)
    y_preds = model.predict(x_test)
    auc = model.score(y_test, y_preds)
    print('logistic model auc:')
    print(auc)





