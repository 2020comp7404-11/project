import pandas as pd
from sklearn.model_selection import train_test_split

def read_data():
    target_col = ['SeriousDlqin2yrs']
    # ignore the index column
    df = pd.read_csv('GiveMeSomeCredit/cs-training.csv', index_col=False).iloc[:,1:]
    x, y = df.drop(target_col, axis=1), df[target_col]
    # fillna with zero
    x.fillna(0, inplace=True)
    # generate categorical data using binning method
    origin_col = ['RevolvingUtilizationOfUnsecuredLines', 'age', 'DebtRatio', 'MonthlyIncome']
    bin_num = 6
    for col in origin_col:
        x[col + '_binning'] = pd.qcut(x[col], bin_num, labels=False, duplicates='drop').values
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1234)
    return x_train, x_test, y_train, y_test
