import pandas as pd
import quandl, math
import numpy as np
from sklearn import preprocessing, model_selection
from sklearn.linear_model import LinearRegression

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

X = np.array(df.drop(['Saida'], 1))
X = preprocessing.scale(X)

y = np.array(df['Saida'])

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = 0.2)

classifier = LinearRegression()
classifier.fit(X_train, y_train)

accuracy = classifier.score(X_test, y_test)

print(accuracy)