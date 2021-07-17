from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import RFE
import numpy as np
import pandas as pd

df = pd.read_csv("https://projects.fivethirtyeight.com/soccer-api/club/spi_matches.csv").dropna()
feat_mask = df.dtypes == object
X_final = df.loc[:,~feat_mask]

df['result'] = df['score1'] - df['score2']

results = []

for i in df['result']:
    if i > 0:
        results.append("HOME")
    elif i < 0:
        results.append("AWAY")
    else:
        results.append("DRAW")

y_final = results

X_train_test, X_test_test, y_train_test, y_test_test = train_test_split(X_final, y_final, test_size=0.2, random_state=6)

rf_test = RandomForestClassifier()
rf_test.fit(X_train_test, y_train_test)
rf_test_predictions = rf_test.predict(X_test_test)
test_accuracy = np.round(accuracy_score(y_test_test, rf_test_predictions),3) * 100
print("Untuned Accuracy: {}%".format(test_accuracy))

feat_mask = rf_test.feature_importances_ >= 0.2
X_new = X_final.loc[:,feat_mask]
X_train, X_test, y_train, y_test = train_test_split(X_new, y_final, test_size=0.2, random_state=6)
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
predictions = rf.predict(X_test)
accuracy = np.round(accuracy_score(y_test, predictions),3) * 100
print("Mask Test Accuracy: {}%".format(accuracy))

mask_dict_form = dict(zip(X_final.columns, rf.feature_importances_))
print("Mask Feature Dictionary:", mask_dict_form)

X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=0.2, random_state=6)

rf = RandomForestClassifier()

rfe = RFE(estimator=rf, n_features_to_select=2, step=2, verbose=1)
rfe.fit(X_train, y_train)
rfe_mask = rfe.support_
X_final = X_final.loc[:,rfe_mask]
X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=0.2, random_state=6)
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
accuracy = np.round(accuracy_score(y_test, predictions),3) * 100
print("RFE Test Accuracy: {}%".format(accuracy))