import pickle
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, LSTM, Dense

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


#Model
model = Sequential()

model.add(Embedding(input_dim = vocab_size, output_dim = 64, input_shape = (max_length,)))
model.add(LSTM(64))
model.add(Dense(vocab_size))

print('Compiling...')
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print('Training...')
model.fit(epochs=10, steps_per_epoch=30)

accuracy = model.evaluate(x_test, y_test)

for a in accuracy:
    print(a)
