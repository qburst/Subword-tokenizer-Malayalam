from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf

from sklearn.model_selection import train_test_split

import numpy as np
import os
import io
import time, datetime
import pickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_path')
args = parser.parse_args()

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
    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding='post')
    
    return tensor, lang_tokenizer


def load_dataset(path, num_examples=None):
    inp_lang, targ_lang, targ_lang2 = create_dataset(path, num_examples)

    input_tensor, inp_lang_tokenizer = tokenize(inp_lang)
    target_tensor, targ_lang_tokenizer = tokenize(targ_lang)
    target_tensor2, targ_lang_tokenizer2 = tokenize(targ_lang2)

    return input_tensor, target_tensor, target_tensor2, inp_lang_tokenizer, targ_lang_tokenizer,targ_lang_tokenizer2


num_examples = 200000
input_tensor, target_tensor, target_tensor2, inp_lang, targ_lang, targ_lang2 = load_dataset(args.data_path, num_examples)
max_length_targ, max_length_inp = max_length(target_tensor), max_length(input_tensor)
input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val, target_tensor_train2, target_tensor_val2 = train_test_split(input_tensor, target_tensor, target_tensor2, test_size=0.2)

with open('inp_lang.pkl', 'wb') as f:
    pickle.dump(inp_lang, f)

with open('targ_lang.pkl', 'wb') as f:
    pickle.dump(targ_lang, f)

BUFFER_SIZE = len(input_tensor_train)
BATCH_SIZE = 64
steps_per_epoch = len(input_tensor_train)//BATCH_SIZE
embedding_dim = 128
units = 512
vocab_inp_size = len(inp_lang.word_index)+1
vocab_tar_size = len(targ_lang.word_index)+1
vocab_tar_size2 = len(targ_lang2.word_index)+1

dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train)).shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
test_dataset = tf.data.Dataset.from_tensor_slices((input_tensor_val, target_tensor_val)).batch(BATCH_SIZE)

dataset2 = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train2)).shuffle(BUFFER_SIZE)
dataset2 = dataset2.batch(BATCH_SIZE, drop_remainder=True)
test_dataset2 = tf.data.Dataset.from_tensor_slices((input_tensor_val, target_tensor_val2)).batch(BATCH_SIZE)


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
        self.lstm = tf.keras.layers.LSTM(self.dec_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.fc = tf.keras.layers.Dense(vocab_size)

        self.attention = attention

    def call(self, x, hidden, enc_output):
        context_vector, attention_weights = self.attention(hidden, enc_output)

        x = self.embedding(x)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        output, state, _ = self.lstm(x)
        output = tf.reshape(output, (-1, output.shape[2]))
        x = self.fc(output)

        return x, state, attention_weights


attention = BahdanauAttention(units)
decoder = Decoder(vocab_tar_size, embedding_dim, units, BATCH_SIZE, attention)
decoder2 = Decoder(vocab_tar_size2, embedding_dim, units, BATCH_SIZE, attention)


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

train_log_dir2 = 'logs/gradient_tape/' + current_time + '/train2'
test_log_dir2 = 'logs/gradient_tape/' + current_time + '/test2'
train_summary_writer2 = tf.summary.create_file_writer(train_log_dir2)
test_summary_writer2 = tf.summary.create_file_writer(test_log_dir2)


@tf.function
def train_step(inp, targ, enc_hidden):
    loss = 0
    accuracy = 0
    with tf.GradientTape() as tape:
        enc_output, enc_hidden = encoder(inp, enc_hidden)

        dec_hidden = enc_hidden
        dec_input = tf.expand_dims([targ_lang2.word_index['<']] * BATCH_SIZE, 1)

        for t in range(1, targ.shape[1]):
            predictions, dec_hidden, _ = decoder2(dec_input, dec_hidden, enc_output)
            loss += loss_function(targ[:, t], predictions)
            accuracy += train_accuracy(targ[:, t], predictions)
            dec_input = tf.expand_dims(targ[:, t], 1)
    batch_loss = (loss / int(targ.shape[1]))
    batch_accuracy = (accuracy/int(targ.shape[1]))
    
    variables = encoder.trainable_variables + decoder2.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))

    return batch_loss, batch_accuracy


@tf.function
def test_step(inp, targ, enc_hidden):
    loss = 0
    accuracy = 0

    enc_output, enc_hidden = encoder(inp, enc_hidden)
    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([targ_lang2.word_index['<']] * BATCH_SIZE, 1)

    for t in range(1, targ.shape[1]):
        predictions, dec_hidden, _ = decoder2(dec_input, dec_hidden, enc_output)
        loss += loss_function(targ[:, t], predictions)
        accuracy +=test_accuracy(targ[:, t], predictions)
        dec_input = tf.expand_dims(targ[:, t], 1)

    batch_loss = (loss/int(targ.shape[1]))
    batch_accuracy = (accuracy/int(targ.shape[1]))

    return batch_loss, batch_accuracy


EPOCHS = 25

for epoch in range(0, EPOCHS):
    start = time.time()

    enc_hidden = encoder.initialize_hidden_state()
    total_loss = 0
    total_accuracy = 0
    total_loss_test = 0
    total_accuracy_test = 0

    for (batch, (inp, targ)) in enumerate(dataset2.take(steps_per_epoch)):
        batch_loss, batch_accuracy = train_step(inp, targ, enc_hidden)
        total_loss += batch_loss
        total_accuracy += batch_accuracy
        
    total_accuracy = total_accuracy/steps_per_epoch
    total_loss = total_loss/steps_per_epoch

    with train_summary_writer.as_default():
        tf.summary.scalar('loss', total_loss, step=epoch)
        tf.summary.scalar('accuracy', total_accuracy, step=epoch)


    for (batch, (inp, targ)) in enumerate(test_dataset2.take(steps_per_epoch_test)):
        batch_loss, batch_accuracy = test_step(inp, targ, enc_hidden)
        total_loss_test += batch_loss
        total_accuracy_test += batch_accuracy

    total_accuracy_test = total_accuracy_test/steps_per_epoch_test
    total_loss_test = total_loss_test/steps_per_epoch_test

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
    
    
@tf.function
def train_step2(inp, targ, enc_hidden):
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
def test_step2(inp, targ, enc_hidden):
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


checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

for epoch in range(0, EPOCHS):
    start = time.time()

    enc_hidden = encoder.initialize_hidden_state()
    total_loss = 0
    total_accuracy = 0
    total_loss_test = 0
    total_accuracy_test = 0

    for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):
        batch_loss, batch_accuracy = train_step2(inp, targ, enc_hidden)
        total_loss += batch_loss
        total_accuracy += batch_accuracy

    total_loss = total_loss/steps_per_epoch
    total_accuracy = total_accuracy/steps_per_epoch

    with train_summary_writer2.as_default():
        tf.summary.scalar('loss', total_loss, step=epoch)
        tf.summary.scalar('accuracy', total_accuracy, step=epoch)

    for (batch, (inp, targ)) in enumerate(test_dataset.take(steps_per_epoch_test)):
        batch_loss, batch_accuracy = test_step2(inp, targ, enc_hidden)
        total_loss_test += batch_loss
        total_accuracy_test += batch_accuracy

    total_loss_test = total_loss_test/steps_per_epoch_test
    total_accuracy_test = total_accuracy_test/steps_per_epoch_test

    with test_summary_writer2.as_default():
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

