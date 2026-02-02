import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler



data = pd.read_csv('Data/CW1_train.csv')

def getCorrelationMatrix(data):
    co_mtx = data.corr(numeric_only=True)
    print(co_mtx)
    sns.heatmap(co_mtx, cmap="YlGnBu", annot=True, fmt=".2f")
    plt.show()


def plotDepthvsOutcome(data):
    sns.scatterplot(x='depth', y='outcome', data=data)
    plt.show()


def getMutualInfo(data):
    X = data.drop(columns=['outcome'])
    y = data['outcome']

    X_ohe = pd.get_dummies(
        X,
        columns=['cut', 'color', 'clarity'],
        drop_first=True
    )
    discrete_mask = X_ohe.dtypes == 'uint8'
    #Scale continuous features (important for kNN MI)
    continuous_cols = X_ohe.columns[~discrete_mask]
    scaler = StandardScaler()
    X_ohe[continuous_cols] = scaler.fit_transform(X_ohe[continuous_cols])

    mi = mutual_info_regression(
        X_ohe,
        y,
        discrete_features=discrete_mask,
        n_neighbors=10,
        random_state=0
    )
    mi_series = pd.Series(mi, index=X_ohe.columns).sort_values(ascending=False)
    return mi_series

def testWithBaselineModel(data):
    pass

#plotDepthvsOutcome(data)
getCorrelationMatrix(data)
#print(getMutualInfo(data))