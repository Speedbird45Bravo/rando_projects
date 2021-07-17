from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd

df = pd.read_csv("https://projects.fivethirtyeight.com/soccer-api/club/spi_matches.csv").dropna().reset_index(drop=True)

num_mask = df.dtypes == object
X_num = df.loc[:,~num_mask]

# First, we need to generate comparisons between the performance metrics of the home and away team.
# Usually margins like these would be generated as absolute values, but since we want to know how the teams stack up relative to one another,
# we need negative values to clearly differentiate the teams, so we're just subtracing the away performance from the home performance.
df["result"] = df["score1"] - df["score2"]
df["proj_score"] = df["proj_score1"] - df["proj_score2"]

results = []
projs = []

def convert(array, list_to_return):
  array = df[array]
  for x in range(len(array)):
    x = array[x]
    if x > 0:
      list_to_return.append("HOME")
    elif x < 0:
      list_to_return.append("AWAY")
    else:
      list_to_return.append("DRAW")

result_list = convert("result", results)
proj_list = convert("proj_score", projs)

df['result'] = results
df['proj_score'] = projs

i_list = []
master_result = []

for i in range(len(df)):
    i1 = df["result"][i]
    i2 = df["proj_score"][i]
    i_list.append([i1, i2])

for i in range(len(i_list)):
    i = i_list[i]
    j = i.count(i[0])==len(i)
    if j == True:
        master_result.append("UNANIMOUS")
    else:
        master_result.append("MIXED")

y_final = master_result

X_train, X_test, y_train, y_test = train_test_split(X_num, y_final, test_size=0.2, random_state=6)

rf_test = RandomForestClassifier()
rf_test.fit(X_train, y_train)

rfe = RFE(estimator=rf_test, n_features_to_select=4, step=4, verbose=1)
rfe.fit(X_train, y_train)
feat_mask = rfe.support_
X_final = X_num.loc[:,feat_mask]
X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=0.2, random_state=6)

param_grid = {'n_estimators':[100,200,300,400,500], 'criterion':['entropy','gini'], 'max_depth':[1,2,3,4,5]}
grid = GridSearchCV(estimator=rf_test, param_grid=param_grid, cv=5)
grid.fit(X_train, y_train)

param_list = list(grid.best_params_.values())

rf = RandomForestClassifier(n_estimators=param_list[2], max_depth=param_list[1], criterion=param_list[0])
rf.fit(X_train, y_train)
predictions = rf.predict(X_test)

accuracy = np.round(accuracy_score(y_test, predictions),3) * 100

print("Test Accuracy Result: {}%".format(accuracy))