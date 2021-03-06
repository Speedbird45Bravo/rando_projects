from feature_engine.imputation import CategoricalImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn_pandas import DataFrameMapper
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd

df = pd.read_csv("https://projects.fivethirtyeight.com/soccer-api/club/spi_matches.csv")
df['result'] = df['score1'] - df['score2']

results = []

for i in df['result']:
    if i > 0:
        results.append("HOME")
    elif i < 0:
        results.append("AWAY")
    else:
        results.append("DRAW")

def encode(array):
    le = LabelEncoder()
    array = le.fit_transform(array)
    return array

df['result'] = encode(results)

X_final = df[['league','team1','team2','spi1','spi2','xg1','xg2','adj_score1','adj_score2','proj_score1','proj_score2','score1','score2','nsxg1','nsxg2']]
y_final = df['result']

categorical_mask = X_final.dtypes == object
cat_columns = X_final.columns[categorical_mask].tolist()
num_columns = X_final.columns[~categorical_mask].tolist()
cat_mapper = DataFrameMapper([([cat_feature], CategoricalImputer()) for cat_feature in cat_columns], input_df=True, df_out=True)
num_mapper = DataFrameMapper([([num_feature], SimpleImputer(missing_values=np.nan, strategy="median")) for num_feature in num_columns], input_df=True, df_out=True)
union = FeatureUnion([("num_mapper", num_mapper), ("cat_mapper", cat_mapper)], verbose=0)

X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=0.2, random_state=46)

class Dictifier(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return pd.DataFrame(X).to_dict("records")

pipe = Pipeline([('feature_union', union),
    ('dictifier', Dictifier()),
    ('vectorizer', DictVectorizer(sort=False)),
    ('model', RandomForestClassifier(n_estimators=100, criterion="entropy"))])

pipe.fit(X_train, y_train)

predictions = pipe.predict(X_test)

accuracy = accuracy_score(y_test, predictions).round(4) * 100

print("Test Accuracy Performance Result: {}%".format(accuracy))