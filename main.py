%load_ext autoreload
%autoreload 2

# Download the data from your GitHub repository
!wget https://raw.githubusercontent.com/yotam-biu/ps9/main/parkinsons.csv -O /content/parkinsons.csv

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

dfA = pd.read_csv('parkinsons.csv')
dfA.dropna()
import matplotlib.pyplot as plt
import seaborn as sns

features= ['HNR', 'RPDE']
output = ['status']

X = dfA[features]
Y = dfA[output]

from sklearn.preprocessing import MinMaxScaler
scaler= MinMaxScaler()
X_scaled= scaler.fit_transform(X)
Y_scaled= scaler.fit_transform(Y)from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=42)
from sklearn.svm import SVC
svm= SVC()
model= svm.fit(X, Y)
from sklearn.metrics import accuracy_score
y_pred = model.predict(X_test)
accuracy = accuracy_score(Y_test, y_pred)
print(f"Accuracy: {accuracy}")
import joblib

joblib.dump(model, 'parkinson.joblib')


