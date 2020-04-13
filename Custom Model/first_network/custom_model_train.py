from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from sklearn.model_selection import train_test_split

import unicodedata
import re
import numpy as np
import os
import io
import time
import argparse, datetime
import pickle


parser = argparse.ArgumentParser()
parser.add_argument('--data_path')
args = parser.parse_args()

path_to_file = args.data_path

def preprocess_sentence(w):
  w = '<' + w + '>'
  return w


def create_dataset(path, num_examples):
  lines = io.open(path, encoding='UTF-8').read().strip().split('\n')
  word_pairs = [[preprocess_sentence(w) for w in l.split('\t')]  for l in lines[:num_examples]]

  return zip(*word_pairs)


def max_length(tensor):
  return max(len(t) for t in tensor)


def tokenize(lang):
  lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(char_level=True)
  lang_tokenizer.fit_on_texts(lang)

  tensor = lang_tokenizer.texts_to_sequences(lang)
  tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor,
                                                         padding='post')

  return tensor, lang_tokenizer


def load_dataset(path, num_examples=None):
  inp_lang, _, targ_lang = create_dataset(path, num_examples)

  input_tensor, inp_lang_tokenizer = tokenize(inp_lang)
  target_tensor, targ_lang_tokenizer = tokenize(targ_lang)

  return input_tensor, target_tensor, inp_lang_tokenizer, targ_lang_tokenizer


num_examples = 10000#175709
input_tensor, target_tensor, inp_lang, targ_lang = load_dataset(path_to_file, num_examples)

max_length_targ, max_length_inp = max_length(target_tensor), max_length(input_tensor)

input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(input_tensor, target_tensor, test_size=0.1)


BUFFER_SIZE = len(input_tensor_train)
BATCH_SIZE = 64
steps_per_epoch = len(input_tensor_train)//BATCH_SIZE
steps_per_epoch_test = len(input_tensor_val)//BATCH_SIZE
embedding_dim = 256
units = 1024
EPOCHS = 25
vocab_inp_size = len(inp_lang.word_index)+1
vocab_tar_size = len(targ_lang.word_index)+1

dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train)).shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
test_dataset = tf.data.Dataset.from_tensor_slices((input_tensor_val, target_tensor_val)).batch(BATCH_SIZE)


with open('inp_lang.pkl', 'wb') as f:
    pickle.dump(inp_lang, f)
    
with open('targ_lang.pkl', 'wb') as f:
    pickle.dump(targ_lang, f)


class Encoder(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
    super(Encoder, self).__init__()
    self.batch_sz = batch_sz
    self.enc_units = enc_units
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.lstm = tf.keras.layers.LSTM(self.enc_units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')

  def call(self, x, hidden):
    x = self.embedding(x)
    output, state, _ = self.lstm(x)
    return output, state

  def initialize_hidden_state(self):
    return tf.zeros((self.batch_sz, self.enc_units))


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
  def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
    super(Decoder, self).__init__()
    self.batch_sz = batch_sz
    self.dec_units = dec_units
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.lstm = tf.keras.layers.LSTM(self.dec_units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')
    self.fc = tf.keras.layers.Dense(vocab_size)

    self.attention = BahdanauAttention(self.dec_units)

  def call(self, x, hidden, enc_output):
    context_vector, attention_weights = self.attention(hidden, enc_output)

    x = self.embedding(x)
    x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

    output, state, _ = self.lstm(x)
    output = tf.reshape(output, (-1, output.shape[2]))
    x = self.fc(output)

    return x, state, attention_weights


encoder = Encoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE)
decoder = Decoder(vocab_tar_size, embedding_dim, units, BATCH_SIZE)


optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')

def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)



checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                 encoder=encoder,
                                 decoder=decoder)



train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('train_accuracy')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('test_accuracy')


current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
test_log_dir = 'logs/gradient_tape/' + current_time + '/test'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
test_summary_writer = tf.summary.create_file_writer(test_log_dir)



@tf.function
def train_step(inp, targ, enc_hidden):
    loss = 0
    accuracy = 0
    with tf.GradientTape() as tape:
        enc_output, enc_hidden = encoder(inp, enc_hidden)
        dec_hidden = enc_hidden
        dec_input = tf.expand_dims([targ_lang.word_index['<']] * BATCH_SIZE, 1)

        for t in range(1, targ.shape[1]):
            predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)
            loss += loss_function(targ[:, t], predictions)
            accuracy += train_accuracy(targ[:, t], predictions)
            dec_input = tf.expand_dims(targ[:, t], 1)

    batch_loss = (loss / int(targ.shape[1]))
    batch_accuracy = (accuracy/int(targ.shape[1]))
    
    variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))

    return batch_loss, batch_accuracy


@tf.function
def test_step(inp, targ, enc_hidden):
    loss = 0
    accuracy = 0

    enc_output, enc_hidden = encoder(inp, enc_hidden)
    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([targ_lang.word_index['<']] * BATCH_SIZE, 1)

    for t in range(1, targ.shape[1]):
        predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)
        loss += loss_function(targ[:, t], predictions)
        accuracy +=test_accuracy(targ[:, t], predictions)
        dec_input = tf.expand_dims(targ[:, t], 1)

    batch_loss = (loss/int(targ.shape[1]))
    batch_accuracy = (accuracy/int(targ.shape[1]))

    return batch_loss, batch_accuracy


for epoch in range(EPOCHS):
    start = time.time()

    enc_hidden = encoder.initialize_hidden_state()
    total_loss = 0
    total_accuracy = 0
    total_loss_test = 0
    total_accuracy_test = 0

    for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):
        batch_loss, batch_accuracy = train_step(inp, targ, enc_hidden)
        total_loss += batch_loss
        total_accuracy += batch_accuracy

    
    total_loss = total_loss/steps_per_epoch
    total_accuracy = total_accuracy/steps_per_epoch

    with train_summary_writer.as_default():
        tf.summary.scalar('loss', total_loss, step=epoch)
        tf.summary.scalar('accuracy', total_accuracy, step=epoch)


    for (batch, (inp, targ)) in enumerate(test_dataset.take(steps_per_epoch_test)):
        batch_loss, batch_accuracy = test_step(inp, targ, enc_hidden)
        total_loss_test += batch_loss
        total_accuracy_test += batch_accuracy

    total_loss_test = total_loss_test/steps_per_epoch_test
    total_accuracy_test = total_accuracy_test/steps_per_epoch_test

    with test_summary_writer.as_default():
        tf.summary.scalar('loss', total_loss_test, step=epoch)
        tf.summary.scalar('accuracy', total_accuracy_test, step=epoch)
    if epoch == 0:
        validation_loss = total_loss_test


    if total_loss_test <= validation_loss:
        validation_loss = total_loss_test
        checkpoint.save(file_prefix = checkpoint_prefix)

    print('Epoch {} Loss {:.4f} Accuracy {:.4f} Val_loss {:.4f} \
        Val_acc {:.4f} Time {:.4f}'.format(epoch + 1,
                                        total_loss,
                                        total_accuracy,
                                        total_loss_test,
                                        total_accuracy_test,
                                        time.time() - start))        
