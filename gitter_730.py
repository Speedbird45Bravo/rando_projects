from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

df = pd.read_csv("game_results.csv")
df = df.drop("v", axis=1)
df['margin'] = df['H'] - df['A']

def encode(array):
    le = LabelEncoder()
    array = array.astype(str)
    array = le.fit_transform(array)
    return array

results = []

for i in df['margin']:
    if i > 0:
        results.append("HOME")
    elif i < 0:
        results.append("AWAY")
    else:
        results.append("DRAW")

df['margin'] = results

game_results = []

for i in range(len(df)):
    i_h = df['Home Team'][i]
    i_a = df['Away Team'][i]
    i_m = df['margin'][i]
    if (i_h == "Liverpool") & (i_m == "HOME"):
        game_results.append("LIVERPOOL HOME")
    elif (i_h == "Manchester United") & (i_m == "HOME"):
        game_results.append("UNITED HOME")
    elif (i_h == "Liverpool") & (i_m == "AWAY"):
        game_results.append("LIVERPOOL AWAY")
    elif (i_h == "Manchester United") & (i_m == "AWAY"):
        game_results.append("UNITED AWAY")
    else:
        game_results.append("DRAW")

df['Result'] = game_results

df['Date'] = pd.to_datetime(df['Date'])

df['Day'] = df['Date'].apply(lambda x: x.day).astype(int)
df['Month'] = df['Date'].apply(lambda x: x.month).astype(int)
df['Year'] = df['Date'].apply(lambda x: x.year).astype(int)
df['Weekday'] = df['Date'].apply(lambda x: x.weekday())
df['Winning Manager'] = encode(df['Winning Manager'])
df['Losing Manager'] = encode(df['Losing Manager'])
df['Weekday'] = df['Weekday'].apply((lambda x: df['Date'][x].strftime("%A")))
df['Weekday'] = encode(df['Weekday'])
df['Home Team'] = encode(df['Home Team'].astype(str))
df['Away Team'] = encode(df['Away Team'].astype(str))
df['Competition'] = encode(df['Competition'].astype(str))
df['Home Team Scorer #1'] = encode(df['Home Team Scorer #1'])
df['Home Team Scorer #2'] = encode(df['Home Team Scorer #2'])
df['Home Team Scorer #3'] = encode(df['Home Team Scorer #3'])
df['Away Team Scorer #1'] = encode(df['Away Team Scorer #1'])
df['Away Team Scorer #2'] = encode(df['Away Team Scorer #2'])
df['Away Team Scorer #3'] = encode(df['Away Team Scorer #3'])
df['Away Team Scorer #4'] = encode(df['Away Team Scorer #4'])

df = df.drop("Date", axis=1)

for col in df.columns:
    df[col] = df[col].fillna(0)

num_cols = df.dtypes != object

X_final = df.loc[:,num_cols]
X_final = X_final.drop(['H','A'], axis=1)

y_final = df['Result']

y_final = encode(y_final)

X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=0.2, random_state=6)

model = RandomForestClassifier()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
pre_acc_score = accuracy_score(y_test, predictions).round(3) * 100
print("Untuned Accuracy: {}%".format(pre_acc_score))

feature_mask = model.feature_importances_ >= 0.1

X_featured = X_final.loc[:,feature_mask]
X_train, X_test, y_train, y_test = train_test_split(X_featured, y_final, stratify=y_final, test_size=0.2, random_state=6)

model = RandomForestClassifier()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
amended_accuracy = np.round(accuracy_score(y_test, predictions),3) * 100
print("Feature Accuracy: {}%".format(amended_accuracy))