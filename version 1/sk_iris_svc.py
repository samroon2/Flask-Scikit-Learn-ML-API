from sklearn import datasets
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
import pandas as pd

#load data from sklearn
iris = datasets.load_iris()

#Create a pandas data frame
df = pd.DataFrame(iris.data)
df.columns = iris.feature_names
df['iris_type'] = iris.target

#Our data is already split when provided by sklearn, asign X, y
X_iris = iris.data
y_iris = iris.target

#split our data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_iris, y_iris, test_size=0.7, random_state=0)

clf = SVC()

#fit our training data to the model
clf.fit(X_train, y_train)

#a quick review of it's performance
print("Train set score: {:2f}".format(clf.score(X_train, y_train)))
y_pred = clf.predict(X_test)
print("Test set score: {:2f}".format(clf.score(X_test, y_test)))

#now serialize our model - to save it in its current trained state - pickle it.
joblib.dump(clf, 'irisclassifier.pkl')