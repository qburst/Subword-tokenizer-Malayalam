from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.model_selection import train_test_split

import unicodedata
import re
import numpy as np
import os
import io
import time, datetime
import pickle

def preprocess_sentence(w):
    w = '<' + w + '>'
    return w

BUFFER_SIZE = 160000
BATCH_SIZE = 64
embedding_dim = 128
units = 512
vocab_inp_size = 74
vocab_tar_size = 75
vocab_tar_size2 = 5
max_length_inp = 35
max_length_targ = 36

with open('inp_lang.pkl', 'rb') as f:
    inp_lang = pickle.load(f)

with open('targ_lang.pkl', 'rb') as f:
    targ_lang = pickle.load(f)


class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.LSTM(self.enc_units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')

    def call(self, x, hidden):
        x = self.embedding(x)
        output, state, _ = self.gru(x)
        return output, state

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.enc_units))

encoder = Encoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE)


class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values):
        query_with_time_axis = tf.expand_dims(query, 1)

        score = self.V(tf.nn.tanh(
            self.W1(query_with_time_axis) + self.W2(values)))

        attention_weights = tf.nn.softmax(score, axis=1)

        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights


class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz, attention):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.LSTM(self.dec_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.fc = tf.keras.layers.Dense(vocab_size)

        self.attention = attention
    def call(self, x, hidden, enc_output):
        context_vector, attention_weights = self.attention(hidden, enc_output)

        x = self.embedding(x)

        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        output, state, _ = self.gru(x)
        output = tf.reshape(output, (-1, output.shape[2]))
        x = self.fc(output)

        return x, state, attention_weights


attention = BahdanauAttention(units)
decoder = Decoder(vocab_tar_size, embedding_dim, units, BATCH_SIZE, attention)
decoder2 = Decoder(vocab_tar_size2, embedding_dim, units, BATCH_SIZE, attention)



optimizer = tf.keras.optimizers.Adam()


checkpoint_dir = './model_checkpoint'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                 encoder=encoder,
                                 decoder=decoder)


checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))


def tokenize(sentence):
    sentence = preprocess_sentence(sentence)

    inputs = inp_lang.texts_to_sequences([sentence])
    inputs = tf.keras.preprocessing.sequence.pad_sequences(inputs,
                                                         maxlen=max_length_inp,
                                                         padding='post')
    inputs = tf.convert_to_tensor(inputs)

    result = ''

    hidden = [tf.zeros((1, units))]
    enc_out, enc_hidden = encoder(inputs, hidden)

    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([targ_lang.word_index['<']], 0)

    for t in range(max_length_targ):
        predictions, dec_hidden, attention_weights = decoder(dec_input,
                                                         dec_hidden,
                                                         enc_out)


        predicted_id = tf.argmax(predictions[0]).numpy()

        result += targ_lang.index_word[predicted_id]

        if targ_lang.index_word[predicted_id] == '>':
            return result[:-1].replace('+', ' + ')

        dec_input = tf.expand_dims([predicted_id], 0)

    return result.replace('+', ' + ')


if __name__=='__main__':
    if(sys.argv[1]):
        print(tokenize(sys.argv[1]))
