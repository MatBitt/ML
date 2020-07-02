import pandas as pd
import quandl, math
import numpy as np
from sklearn import preprocessing, neighbors
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Regressao linear

df = quandl.get('WIKI/GOOGL')
df = df[['Adj. Open', 'Adj. Close', 'Adj. High', 'Adj. Low', 'Adj. Volume',]]
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low']) / df['Adj. Low'] * 100
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100

df = df[['HL_PCT', 'PCT_change', 'Adj. Volume', 'Adj. Close']]

forecast_col = 'Adj. Close'  
df.fillna(-99999, inplace=True)

forecast_out = int(math.ceil(0.01*len(df)))

df['Saida'] = df[forecast_col].shift(-forecast_out)
df.dropna(inplace=True)

LR_X = np.array(df.drop(['Saida'], 1))
LR_X = preprocessing.scale(LR_X)

LR_y = np.array(df['Saida'])

LR_X_train, LR_X_test, LR_y_train, LR_y_test = train_test_split(LR_X, LR_y, test_size = 0.2)

LR_classifier = LinearRegression()
LR_classifier.fit(LR_X_train, LR_y_train)

LR_accuracy = LR_classifier.score(LR_X_test, LR_y_test)

print('Accuracy for linear regression :', LR_accuracy)

# K nearest neighbors

data = pd.read_csv('data.txt')

data.replace('?', -99999, inplace=True)
data.drop(['id'], 1, inplace=True)

KN_X = np.array(data.drop(['Saida'], 1))
KN_y = np.array(data['Saida'])

KN_X_train, KN_X_test, KN_y_train, KN_y_test = train_test_split(KN_X, KN_y, test_size = 0.2)

KN_classifier = neighbors.KNeighborsClassifier()
KN_classifier.fit(KN_X_train, KN_y_train)

KN_accuracy = KN_classifier.score(KN_X_test, KN_y_test)

print('Accuracy for K nearest neighbors :', KN_accuracy)



