from keras.callbacks import LambdaCallback
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import numpy as np
import random
import sys
import io


HIDDEN_DIM = 10
LAYER_NUM = 2

model = Sequential()
model.add(LSTM(HIDDEN_DIM, input_shape=(None, VOCAB_SIZE), return_sequence=True))

for i in range(LAYER_NUM - 1):
    model.add(LSTM(HIDDEN_DIM, return_sequences=True))

model.add(TimeDistributed(Dense(VOCAB_SIZE)))

model.add(Activation("softmax"))
model.compile(loss="categorical_crossentropy", optimizer="rmsprop")
