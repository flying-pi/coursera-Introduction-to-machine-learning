import re

import pandas

data = pandas.read_csv(
    'train.csv',
    index_col='PassengerId')
sex = data['Sex']
maleCount = sex[data['Sex'] == 'male'].size
print(maleCount, sex.size - maleCount)

survived = data['Survived']
survivedCount = survived[data['Survived'] == 1].size
print("%.2f" % (survivedCount / survived.size * 100.00))

classes = data['Pclass']
classesCount = classes[data['Pclass'] == 1].size
print("%.2f" % (classesCount / classes.size * 100.00))

ages = data['Age']
print("%.2f" % ages.mean(), ages.median())

print("%.2f" % data['SibSp'].corr(data['Parch']))


def process_function(x: str):
    m = re.search("(Miss|Mrs)\\. (.+?)( |\\.|$)", x)
    if m:
        return m.groups()[-2].strip().strip(".").strip(",").strip("(")
    else:
        return x.strip()


famaleNames = data['Name'][data['Sex'] == 'female']
famaleNames = famaleNames.apply(process_function)
print(famaleNames.value_counts())

