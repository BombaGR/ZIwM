import numpy
from pandas import read_csv
from sklearn.decomposition import PCA
# load data
names = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', 'class']
dataframe = read_csv("bialaczka.csv", names=names)
array = dataframe.values
print(array)
X = array[1:, 0:20]
Y = array[1:, 20]
# feature extraction
pca = PCA(n_components=3)
fit = pca.fit(X)
# summarize components
print("Explained Variance: %s") % fit.explained_variance_ratio_
print(fit.components_)