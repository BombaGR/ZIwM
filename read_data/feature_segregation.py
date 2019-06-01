from pandas import read_csv
from sklearn.ensemble import ExtraTreesClassifier
# load data
names = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', 'class']
dataframe = read_csv("bialaczka.csv", names=names)
array = dataframe.values
X = array[1:, 3:23]
Y = array[1:, 20]
# feature extraction
model = ExtraTreesClassifier()
model.fit(X, Y)
print(model.feature_importances_)