import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from deepctr.feature_column import SparseFeat, DenseFeat,get_feature_names
import matplotlib.pyplot as plt

def read_data():
    target_col = ['SeriousDlqin2yrs']
    # ignore the index column
    df = pd.read_csv('GiveMeSomeCredit/cs-training.csv', index_col=False).iloc[:,1:]
    x, y = df.drop(target_col, axis=1), df[target_col]
    # fillna with zero
    x.fillna(0, inplace=True)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1234)
    return x_train, x_test, y_train, y_test

def read_data_as_model():
    data = pd.read_csv('GiveMeSomeCredit/cs-training.csv')
    sparse_features = ['NumberOfTime30-59DaysPastDueNotWorse', 'NumberOfTimes90DaysLate', 'NumberOfTime60-89DaysPastDueNotWorse', 'NumberOfDependents']
    dense_features = ['RevolvingUtilizationOfUnsecuredLines', 'age', 'DebtRatio', 'MonthlyIncome', 'NumberOfOpenCreditLinesAndLoans', 'NumberRealEstateLoansOrLines']

    data[sparse_features] = data[sparse_features].fillna(0)
    data[dense_features] = data[dense_features].fillna(data[dense_features].mean())
    target = ['SeriousDlqin2yrs']

    # 1.Label Encoding for sparse features,and do simple Transformation for dense features
    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])
    mms = MinMaxScaler(feature_range=(0, 1))
    data[dense_features] = mms.fit_transform(data[dense_features])

    # 2.count #unique features for each sparse field,and record dense feature field name

    fixlen_feature_columns = [SparseFeat(feat, vocabulary_size=data[feat].nunique(),embedding_dim=4 )
                           for i,feat in enumerate(sparse_features)] + [DenseFeat(feat, 1,)
                          for feat in dense_features]

    dnn_feature_columns = fixlen_feature_columns
    linear_feature_columns = fixlen_feature_columns
    feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

    # 3.generate input data for model

    train, test = train_test_split(data, test_size=0.2, random_state=1234)
    train_model_input = {name:train[name] for name in feature_names}
    test_model_input = {name:test[name] for name in feature_names}

    return train, test, train_model_input, test_model_input, dnn_feature_columns, linear_feature_columns, feature_names, target

def plot_roc(fpr_list, tpr_list, auc_list, name_list):
    plt.figure(figsize=(6, 6))
    plt.title('Validation ROC')
    for i in range(len(fpr_list)):
        plt.plot(fpr_list[i], tpr_list[i], label='%s AUC = %s' % (name_list[i], auc_list[i]))

    plt.plot([0, 1], [0, 1], 'r--')
    plt.legend(loc='lower right')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()