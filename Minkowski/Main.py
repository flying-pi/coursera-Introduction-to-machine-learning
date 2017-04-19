from numpy import linspace
from numpy.ma.core import mean
from sklearn.cross_validation import cross_val_score
from sklearn.datasets import load_boston
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import scale

bostonDataset = load_boston();
target = bostonDataset.target
data = scale(load_boston().data)

maxValue = -100000
maxP = -1;
for i in linspace(start=1, stop=10, num=200):
    regressionFun = KNeighborsRegressor(n_neighbors=5, weights='distance', metric='minkowski');
    regressionFun.p = i
    score = cross_val_score(estimator=regressionFun,
                            cv=KFold(n_splits=5, random_state=42, shuffle=True).split(data),
                            X=data, y=target, scoring='neg_mean_squared_error')
    meanScore = mean(score)
    if meanScore > maxValue:
        maxValue = meanScore
        maxP = i
    print(i, meanScore)

print(maxP, maxValue)
