from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd

df = pd.read_csv("https://projects.fivethirtyeight.com/soccer-api/club/spi_matches.csv").dropna()
df['result'] = df['score1'] - df['score2']

results = []

for i in df['result']:
    if i >= 3:
        results.append("HOME BIG")
    elif i == 2 :
        results.append("HOME DECISIVE")
    elif i == 1:
        results.append("HOME EDGE")
    elif i <= -3:
        results.append("AWAY BIG")
    elif i == -2:
        results.append("AWAY DECISIVE")
    elif i == -1:
        results.append("AWAY EDGE")
    else:
        results.append("DRAW")

df = df.drop("result", axis=1)

y_final = results

int_mask = df.dtypes == int
num_mask = df.dtypes == float
ints = df.loc[:, int_mask]
nums = df.loc[:, num_mask]
X_final = pd.concat([ints, nums], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=0.2, random_state=6)

rf_test = RandomForestClassifier(criterion="gini", n_estimators=300)
rf_test.fit(X_train, y_train)
predictions = rf_test.predict(X_test)

mask = rf_test.feature_importances_ >= 0.2
X_final = X_final.loc[: , mask]
X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=0.2, random_state=6)

wijnaldum = "gini"
rf_final = RandomForestClassifier(criterion=wijnaldum, n_estimators=300)
rf_final.fit(X_train, y_train)
predictions = rf_final.predict(X_test)

accuracy = np.round(accuracy_score(y_test, predictions) * 100, 3)
print("Test Accuracy: {}%".format(accuracy))