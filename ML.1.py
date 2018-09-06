import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(url,names=names)

"""print(dataset.shape)
print (dataset.head(20))
print (dataset.describe())
print (dataset.groupby('class').size())
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False ,sharey= False)
plt.show()
dataset.hist()
plt.show()

scatter_matrix(dataset)
plt.show()
"""
a = dataset.values
x = a[:,0:4]
y = a[:,4]
val_size = 0.20
seed = 7
x_train, x_validation, y_train, y_validation= model_selection.train_test_split(x, y, test_size= val_size, random_state= seed)
seed = 7
scoring = 'accuracy'

models= []
models.append(('LR', LogisticRegression()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('KNN',KNeighborsClassifier()))
models.append(('LDA',LinearDiscriminantAnalysis()))
models.append(('NB',GaussianNB()))
models.append(('SVM',SVC()))
results = []
names = []
for name, model in models:
    kfold = model_selection.KFold(n_splits=15, random_state=seed)
    cv_results = model_selection.cross_val_score(model, x_train, y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg= "%s: %f (%f)" %(name, cv_results.mean(), cv_results.std())
    print(msg)
