from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pandas as pd

pd.options.mode.chained_assignment = None

df = pd.read_csv("fifa21.csv", sep=";")

y_final = df.iloc[:, 4]
to_drop = ["overall","player_id"]
X_init = df.drop(to_drop, axis=1)
mask = X_init.dtypes == object
X_num = X_init.loc[:, ~mask]
X_obj = X_init.loc[:, mask]
le = LabelEncoder()
ss = StandardScaler()

for col in X_num.columns:
    X_num[col].fillna(X_num[col].mean(), inplace=True)

X_scaled = ss.fit_transform(X_num)
X_scaled = pd.DataFrame(X_scaled)

for col in X_obj.columns:
    X_obj[col].fillna(X_obj[col].value_counts().index[0], inplace=True)
    X_obj[col] = le.fit_transform(X_obj[col])

X_final = pd.concat([X_scaled, X_obj], axis=1)

buckets = []

for y in y_final:
    if y >= 90:
        buckets.append("CONSISTENT WORLD CLASS")
    elif (y < 90) & (y >= 85):
        buckets.append("OCCASIONAL WORLD CLASS")
    elif (y < 85) & (y >= 80):
        buckets.append("CONSISTENT PROFESSIONAL")
    elif (y < 75) & (y >= 70):
        buckets.append("OCCASIONAL PROFESSIONAL")
    elif (y < 70) & (y >= 60):
        buckets.append("SEMIPRO")
    else:
        buckets.append("LOWER SEMIPRO")

y_final = le.fit_transform(buckets)

X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=0.2, random_state=6)

model = GradientBoostingClassifier(learning_rate=0.075, n_estimators=500)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions).round(3) * 100
print("Test Accuracy: {}%".format(accuracy))