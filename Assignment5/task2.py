import pickle
import tensorflow as tf
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Embedding, LSTM, Dense
from keras.optimizers import RMSprop

#Loading data
print('Loading data...')
data = pickle.load(open('data/keras-data.pickle','rb'))

x_train = data['x_train']
y_train = data['y_train']
x_test = data['x_test']
y_test = data['y_test']

vocab_size = data['vocab_size']
max_length = data['max_length']

#Padding
print('Padding...')
x_train = pad_sequences(x_train, maxlen=max_length)
x_test = pad_sequences(x_test, maxlen=max_length)

y_train = to_categorical(y_train, num_classes=2)
y_test = to_categorical(y_test, num_classes=2)

#Model
model = Sequential()

model.add(Embedding(input_dim = vocab_size, output_dim = 64))
model.add(LSTM(units = 64, activation = 'relu'))
model.add(Dense(units = 2))

print('Compiling...')
model.compile(loss='categorical_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])

with tf.device('/GPU:0'):
    print('Training...')
    model.fit(x_train, y_train, epochs=3, batch_size = 64)

print('Saved model...')
model.save('models/lstm.h5')

evaluation = model.evaluate(x_test, y_test)

print('Loss: {} | Accuracy: {}'.format(evaluation[0], evaluation[1]))
