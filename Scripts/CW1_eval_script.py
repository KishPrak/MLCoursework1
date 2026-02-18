import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from main import get_pipeline
import pickle

# Set seed
np.random.seed(123)

df_train = pd.read_csv('Data/CW1_train.csv')
df_test = pd.read_csv('Data/CW1_test.csv')

target = "outcome"
features_top8 = ['depth', 'b3', 'b1', 'a1', 'a4', 'a3', 'y', 'price']
categorical_cols = ["cut", "color", "clarity"]
X_train = df_train[features_top8]
y = df_train[target]
X_test = df_test[features_top8]

def prepareTrainDataAndTrainModel():
    pipe = get_pipeline("hybrid", features_top8)
    return pipe.fit(X_train, y)

def prepareTestData(pipe):
    return pipe.predict(X_test)

pipe = prepareTrainDataAndTrainModel()
yhat = prepareTestData(pipe)

out = pd.DataFrame({'yhat': yhat})
out.to_csv('CW1_submission_K23153494.csv', index=False) 

################################################################################

# At test time, we will use the true outcomes
y_tst = pd.read_csv('Data/CW1_test_with_true_outcome.csv') # You do not have access to this

# This is the R^2 function
def r2_fn(yhat):
    eps = y_tst - yhat
    rss = np.sum(eps ** 2)
    tss = np.sum((y_tst - y_tst.mean()) ** 2)
    r2 = 1 - (rss / tss)
    return r2


print(r2_fn(yhat))





