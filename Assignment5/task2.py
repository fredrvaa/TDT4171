import pickle
import tensorflow.keras as keras

#Loading data
data = pickle.load(open('data/sklearn-keras.pickle','rb'))

x_train = data['x_train']
y_train = data['y_train']
x_test = data['x_test']
y_test = data['y_test']