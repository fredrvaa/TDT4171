import pickle
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score    

#Load data
print('Loading data...')
data = pickle.load(open('data/sklearn-data.pickle','rb'))

x_train = data['x_train']
y_train = data['y_train']
x_test = data['x_test']
y_test = data['y_test']

#Hashing
print('Hashing...')
hv = HashingVectorizer(n_features=10, stop_words='english')
x_train = hv.transform(x_train)
x_test =hv.transform(x_test)

#Instantiate classifier
nb_classifier = BernoulliNB()
dt_classifier = DecisionTreeClassifier(max_features=10)

#Training classifier
print('Training BernoulliNB Classifier...')
nb_classifier.fit(x_train, y_train)

print('Training DecisionTree Classifier...')
dt_classifier.fit(x_train, y_train)

#Getting predicted values
nb_pred = nb_classifier.predict(x_test)
dt_pred = dt_classifier.predict(x_test)

#Checking accuracy
print('Accuracy score for BernouilliNB Classifier: {}'.format(accuracy_score(y_test, nb_pred)))
print('Accuracy score for DecisionTree Classifier: {}'.format(accuracy_score(y_test, dt_pred)))


