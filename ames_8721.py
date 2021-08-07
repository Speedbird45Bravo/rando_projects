from sklearn.ensemble import AdaBoostRegressor, BaggingRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None

df = pd.read_csv("https://raw.githubusercontent.com/wblakecannon/ames/master/data/housing.csv")
y_final = df.iloc[:,-1]
df = df.drop(["SalePrice", "Unnamed: 0"], axis=1)

mask = df.dtypes == object

X_obj = df.loc[:,mask]
X_num = df.loc[:,~mask]

for col in X_num.columns:
  X_num[col].fillna(X_num[col].mean(), inplace=True)

for col in X_obj.columns:
  X_obj[col].fillna(X_obj[col].value_counts().index[0], inplace=True)
  le = LabelEncoder()
  X_obj[col] = le.fit_transform(X_obj[col])

X_final = pd.concat([X_obj, X_num], axis=1)
ss = StandardScaler()
X_final = ss.fit_transform(X_final)

X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=0.2, random_state=6)

model = GradientBoostingRegressor()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
test_r2 = np.round(r2_score(y_test, predictions),4)

print("Test R2 Score: {}".format(test_r2))