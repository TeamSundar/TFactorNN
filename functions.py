from Bio import SeqIO
import os
from numpy import argmax
import numpy as np

# Import modules
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Display
from IPython.display import Image, display
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tqdm import tqdm
import os
import tensorflow as tf

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Layer,Input, Dense,LSTM, Dropout, Activation, Concatenate, Flatten, BatchNormalization, Conv1D, MaxPooling1D,Lambda,multiply,Permute,Reshape,RepeatVector
from tensorflow.keras.regularizers import l2,l1
from tensorflow.keras.layers import Bidirectional, TimeDistributed
from tensorflow.keras.optimizers import SGD,Adam,RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint,ReduceLROnPlateau
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K

def onehotEncode(data):
    # define universe of possible input values
    bases = 'ATGC'
    # define a mapping of chars to integers
    char_to_int = dict((c, i) for i, c in enumerate(bases))
    int_to_char = dict((i, c) for i, c in enumerate(bases))
    # integer encode input data
    integer_encoded = [char_to_int[char] for char in data]
    # one hot encode
    onehot_encoded = list()
    for value in integer_encoded:
        letter = [0 for _ in range(len(bases))]
        letter[value] = 1
        onehot_encoded.append(letter)
    #return list(map(list, zip(*onehot_encoded)))
    return(onehot_encoded)

# Encode data
def encode(record):
    a = []
    for i in record:
        data = str(i.seq).upper()
        try:
            a.append(onehotEncode(data))
        except:
            pass
    return a

def build_model():
    # Build model
    sequence_input = Input(shape=(1000,4))

    # Convolutional Layer
    output = Conv1D(320,kernel_size=26,padding="valid",activation="relu")(sequence_input)
    output = MaxPooling1D(pool_size=13, strides=13)(output)
    output = Dropout(0.4)(output)

    #Attention Layer
    attention = Dense(1)(output)
    attention = Permute((2, 1))(attention)
    attention = Activation('softmax')(attention)
    attention = Permute((2, 1))(attention)
    attention = Lambda(lambda x: K.mean(x, axis=2), name='attention',output_shape=(75,))(attention)
    attention = RepeatVector(320)(attention)
    attention = Permute((2,1))(attention)
    output = multiply([output, attention])

    #BiLSTM Layer
    output = Bidirectional(LSTM(320,return_sequences=True))(output)
    output = Dropout(0.4)(output)

    flat_output = Flatten()(output)

    #FC Layer
    FC_output = Dense(640)(flat_output)
    FC_output = Activation('relu')(FC_output)
    FC_output = Dense(256)(FC_output)
    FC_output = Activation('relu')(FC_output)

    #Output Layer
    #classification output
    output  = Dense(1)(FC_output)
    output = Activation('sigmoid')(output)

    # create model with two inputs
    model = Model(inputs=sequence_input, outputs=output)
    print(model.summary())
    return model

def build_model_v2():
    sequence_input = Input(shape=(1000,4))
    esm_input = Input(shape=(768))
    FC_in1 = Activation('relu')(esm_input)

    # Convolutional Layer
    output1 = Conv1D(320,kernel_size=26,padding="valid",activation="relu")(sequence_input)
    output2 = MaxPooling1D(pool_size=13, strides=13)(output1)
    output3 = Dropout(0.4)(output2)

    #Attention Layer
    attention1 = Dense(1)(output3)
    attention2 = Permute((2, 1))(attention1)
    attention3 = Activation('softmax')(attention2)
    attention4 = Permute((2, 1))(attention3)
    attention5 = Lambda(lambda x: K.mean(x, axis=2), name='attention',output_shape=(75,))(attention4)
    attention6 = RepeatVector(320)(attention5)
    attention7 = Permute((2,1))(attention6)
    output4 = multiply([output3, attention7])

    #BiLSTM Layer
    output5 = Bidirectional(LSTM(320,return_sequences=True))(output4)
    output6 = Dropout(0.4)(output5)

    flat_output = Flatten()(output6)

    #Concatenate
    concat = Concatenate()([flat_output, FC_in1])

    #FC Layer
    FC_output1 = Dense(512)(concat)
    #FC_output = Dense(512)(flat_output)
    FC_output2 = Activation('relu')(FC_output1)

    #Output Layer
    #classification output
    final1  = Dense(1)(FC_output2)
    final2 = Activation('sigmoid')(final1)

    # create model with two inputs
    model = Model(inputs=[sequence_input, esm_input], outputs=final2)
    print(model.summary())
    return model
