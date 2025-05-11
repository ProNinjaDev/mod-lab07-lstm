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
import os

with open('src/input.txt', 'r', encoding='utf-8') as file:
    text = file.read()

text_for_processing = text.lower()
text_for_processing = text_for_processing.replace('\n', ' ') # перенос строк
punctuation_to_remove = ['.', ',', '!', '?', ':', ';', '(', ')', '"', '«', '»']
for char in punctuation_to_remove:
     text_for_processing = text_for_processing.replace(char, ' ')

words = text_for_processing.split()
words = [word for word in words if word]

vocabulary = sorted(list(set(words)))


word_to_indices = dict((w, i) for i, w in enumerate(vocabulary))
indices_to_word = dict((i, w) for i, w in enumerate(vocabulary))


max_length = 10
steps = 1
sequences = []
next_words = []

for i in range(0, len(words) - max_length, steps):
    sequences.append(words[i: i + max_length])
    next_words.append(words[i + max_length])


X = np.zeros((len(sequences), max_length, len(vocabulary)), dtype = np.bool_)
y = np.zeros((len(sequences), len(vocabulary)), dtype = np.bool_)

for i, sequence in enumerate(sequences):
    for t, word in enumerate(sequence):
        X[i, t, word_to_indices[word]] = 1
    y[i, word_to_indices[next_words[i]]] = 1

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
    start_index = random.randint(0, len(words) - max_length - 1)
    generated = []
    sequence = words[start_index: start_index + max_length]
    generated.extend(sequence)
    for i in range(length):
            x_pred = np.zeros((1, max_length, len(vocabulary)))
            for t, word in enumerate(sequence):
                x_pred[0, t, word_to_indices[word]] = 1.

            preds = model.predict(x_pred, verbose = 0)[0]
            next_index = sample_index(preds, diversity)
            next_word = indices_to_word[next_index]

            generated.append(next_word)
            sequence = sequence[1:] + [next_word]
    return " ".join(generated)

num_words_generate = 1000
diversity_user = 0.2
generated_text = generate_text(num_words_generate, diversity_user)

output_file_path = os.path.join('result', 'gen.txt')

with open(output_file_path, 'w', encoding='utf-8') as f:
     f.write(generated_text)