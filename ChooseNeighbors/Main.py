import numpy
import pandas
from sklearn.cross_validation import cross_val_score
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import scale

rawData = pandas.read_csv('wine.data', header=None)

category = numpy.array(rawData[0])
rawData.__delitem__(0)
dataList = numpy.array([list(item) for item in rawData.to_records(index=False)])


def optimal_value(data, classes):
    max_value = 0
    max_value_index = -1
    for i in range(1, 51):
        score = cross_val_score(estimator=KNeighborsClassifier(n_neighbors=i),
                                cv=KFold(n_splits=5, random_state=42, shuffle=True).split(data),
                                X=data, y=classes)
        mean_score = numpy.mean(score)
        # print(i, mean_score, "\t", score)
        if mean_score > max_value:
            max_value = mean_score
            max_value_index = i
    return max_value_index, max_value

print(optimal_value(dataList, category))
print(optimal_value(scale(dataList, axis=0), category))
