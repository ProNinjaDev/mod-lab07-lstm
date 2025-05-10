import numpy as np
import tensorflow as tf
from tensorflow import keras

# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Activation
# from tensorflow.keras.layers import LSTM

# from tensorflow.keras.optimizers import RMSprop

# from tensorflow.keras.callbacks import LambdaCallback
# from tensorflow.keras.callbacks import ModelCheckpoint
# from tensorflow.keras.callbacks import ReduceLROnPlateau 
import random
import sys

with open('kolobok.txt', 'r') as file:
    text = file.read()

vocabulary = sorted(list(set(text)))


char_to_indices = dict((c, i) for i, c in enumerate(vocabulary))
indices_to_char = dict((i, c) for i, c in enumerate(vocabulary))


max_length = 10
steps = 1
sentences = []
next_chars = []


for i in range(0, len(text) - max_length, steps):
    sentences.append(text[i: i + max_length])
    next_chars.append(text[i + max_length])


X = np.zeros((len(sentences), max_length, len(vocabulary)), dtype = np.bool_)

y = np.zeros((len(sentences), len(vocabulary)), dtype = np.bool_)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        X[i, t, char_to_indices[char]] = 1
    y[i, char_to_indices[next_chars[i]]] = 1

# Строим LSTM-сеть
model = keras.models.Sequential()
model.add(keras.layers.LSTM(128, input_shape =(max_length, len(vocabulary))))
model.add(keras.layers.Dense(len(vocabulary)))
model.add(keras.layers.Activation('softmax'))
optimizer = keras.optimizers.RMSprop(learning_rate = 0.01)
model.compile(loss ='categorical_crossentropy', optimizer = optimizer)

def sample_index(preds, temperature = 1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

# Обучение LSTM модели
model.fit(X, y, batch_size = 128, epochs = 50)

def generate_text(length, diversity):
    # Случайное начало
    start_index = random.randint(0, len(text) - max_length - 1)
    generated = ''
    sentence = text[start_index: start_index + max_length]
    generated += sentence
    for i in range(length):
            x_pred = np.zeros((1, max_length, len(vocabulary)))
            for t, char in enumerate(sentence):
                x_pred[0, t, char_to_indices[char]] = 1.

            preds = model.predict(x_pred, verbose = 0)[0]
            next_index = sample_index(preds, diversity)
            next_char = indices_to_char[next_index]

            generated += next_char
            sentence = sentence[1:] + next_char
    return generated

print(generate_text(1500, 0.2)) 