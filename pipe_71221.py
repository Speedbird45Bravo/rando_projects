from sklearn_pandas import DataFrameMapper
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import accuracy_score
from feature_engine.imputation import CategoricalImputer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from datetime import datetime
import numpy as np
import pandas as pd

df = pd.read_csv("https://projects.fivethirtyeight.com/soccer-api/club/spi_matches.csv")
df['date'] = pd.to_datetime(df['date'])
today = datetime.today()
df = df[df.date <= today]
df['result'] = df['score1'] - df['score2']

class Dictifier(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return pd.DataFrame(X).to_dict("records")

X_final = df[['league', 'team1', 'team2', 'spi1',
       'spi2', 'prob1', 'prob2', 'probtie', 'proj_score1', 'proj_score2',
       'importance1', 'importance2', 'score1', 'score2', 'xg1', 'xg2', 'nsxg1',
       'nsxg2', 'adj_score1', 'adj_score2']]

results = []

for i in df['result']:
    if i > 0:
        results.append("HOME")
    elif i < 0:
        results.append("AWAY")
    else:
        results.append("DRAW")

df['result'] = results
y_final = df['result']
df = df.drop(["result","season","date","league_id"], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=0.2, random_state=6)

cat_mask = df.dtypes == object
cat_cols = df.columns[cat_mask].tolist()
cat_map = DataFrameMapper([([cat_feature], CategoricalImputer()) for cat_feature in cat_cols], input_df=True, df_out=True)
num_cols = df.columns[~cat_mask].tolist()
num_map = DataFrameMapper([([num_feature], SimpleImputer(missing_values=np.nan, strategy="mean")) for num_feature in num_cols], input_df=True, df_out=True)
union = FeatureUnion([("cat_map", cat_map), ("num_map", num_map)])

pipe = Pipeline([("feature_union", union), ("dictifier", Dictifier()), ("vectorizer", DictVectorizer(sort=False)), ("clf", RandomForestClassifier())])
pipe.fit(X_train, y_train)
predictions = pipe.predict(X_test)
accuracy = accuracy_score(y_test, predictions).round(4) * 100
print("Test Accuracy Result: {}%".format(accuracy))