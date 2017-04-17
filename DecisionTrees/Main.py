import pandas
from sklearn.tree import DecisionTreeClassifier

data = pandas.read_csv(
    'train.csv',
    index_col='PassengerId')

target = data[["Pclass", "Fare", "Age", "Survived"]]
sex = data["Sex"]
sex = sex.apply(lambda x: int(1 if x == 'male' else 0))
target["Sex"] = sex

target = target.dropna()

target_raw = [list(x) for x in target.to_records(index=False)]
survived_raw = list(data["Survived"])

print(target[["Pclass", "Fare", "Age", "Sex"]])

clf = DecisionTreeClassifier(random_state=241)
clf.fit(target[["Pclass", "Fare", "Age", "Sex"]], target["Survived"])

clf.decision_path(target[["Pclass", "Fare", "Age", "Sex"]].irow(5))

print(clf.decision_path(target[["Pclass", "Fare", "Age", "Sex"]].irow(5)))
