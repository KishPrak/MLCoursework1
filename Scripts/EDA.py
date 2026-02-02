import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.inspection import permutation_importance



data = pd.read_csv('Data/CW1_train.csv')

#Initial check if can be solved by linear model
def getCorrelationMatrix(data):
    co_mtx = data.corr(numeric_only=True)
    print(co_mtx)
    sns.heatmap(co_mtx, cmap="YlGnBu", annot=True, fmt=".2f")
    plt.show()

#Checking relationship between target var and the highest correlated feature
def plotDepthvsOutcome(data):
    sns.scatterplot(x='depth', y='outcome', data=data)
    plt.show()

#Since linear didn't yield high results, checking mutual information to see non-linear relationships
def getMutualInfo(data):
    X = data.drop(columns=['outcome'])
    y = data['outcome']

    X_ohe = pd.get_dummies(
        X,
        columns=['cut', 'color', 'clarity'],
        drop_first=True
    )
    discrete_mask = X_ohe.dtypes == 'uint8'
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


#plotDepthvsOutcome(data)
getCorrelationMatrix(data)
#print(getMutualInfo(data))