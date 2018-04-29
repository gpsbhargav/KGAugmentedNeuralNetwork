from __future__ import print_function

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.set_random_seed(1234)
session = tf.Session(graph=tf.get_default_graph(),config=config)

from keras.backend.tensorflow_backend import set_session
set_session(session)


import os
import sys
import tempfile

import numpy as np
import dill as pickle

import keras
from keras.preprocessing.text import Tokenizer
import keras.backend as K
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Input, Embedding, Activation
from keras.layers import CuDNNLSTM
from keras.models import Model,Sequential
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint

np.random.seed(100)

BASE_DIR = '.'
GLOVE_DIR = os.path.join('/home/bhargav/glove6B', 'glove')
#TEXT_DATA_DIR = os.path.join(BASE_DIR, '20_newsgroup')
DATA_DIR = "/home/bhargav/nlu_project/keras_n20/data/Es_Rc"
DOC_PKL = "document_list.pick"
TARGET_PKL = "target_list.pick"

OUTPUT_PATH = './model_outputs/lstm-avg/'

MAX_SEQUENCE_LENGTH = 300
MAX_NUM_WORDS = 20000
EMBEDDING_DIM = 300
LSTM_HIDDEN_SIZE = 200
VALIDATION_SPLIT = 0.1
TEST_SPLIT = 0.2
NUM_EPOCHS = 20
BATCH_SIZE = 256
LEARNING_RATE = 0.001


def pickler(path,pkl_name,obj):
    with open(os.path.join(path, pkl_name), 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def unpickler(path,pkl_name):
    with open(os.path.join(path, pkl_name) ,'rb') as f:
        obj = pickle.load(f)
    return obj



# first, build index mapping words in the embeddings set
# to their embedding vector

print('Indexing word vectors.')

embeddings_index = {}
with open(os.path.join(GLOVE_DIR, 'glove.6B.300d.txt'),encoding='utf8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

print('Found %s word vectors.' % len(embeddings_index))

# second, prepare text samples and their labels
print('Processing text dataset')

texts = []  # list of text samples
labels_index = {}  # dictionary mapping label name to numeric id
labels = []  # list of label ids
#for name in sorted(os.listdir(TEXT_DATA_DIR)):
#    path = os.path.join(TEXT_DATA_DIR, name)
#    if os.path.isdir(path):
#        label_id = len(labels_index)
#        labels_index[name] = label_id
#        for fname in sorted(os.listdir(path)):
#            if fname.isdigit():
#                fpath = os.path.join(path, fname)
#                args = {} if sys.version_info < (3,) else {'encoding': 'latin-1'}
#                with open(fpath, **args) as f:
#                    t = f.read()
#                    i = t.find('\n\n')  # skip header
#                    if 0 < i:
#                        t = t[i:]
#                    texts.append(t[:MAX_SEQUENCE_LENGTH])
#                labels.append(label_id)

raw_docs = unpickler(DATA_DIR,DOC_PKL)
labels = unpickler(DATA_DIR,TARGET_PKL)

for doc in raw_docs:
    texts.append(" ".join(doc.split()[:MAX_SEQUENCE_LENGTH]))


print('Found %s texts.' % len(texts))

# finally, vectorize the text samples into a 2D integer tensor
tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

labels = to_categorical(np.asarray(labels))
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

# split the data into a training set and a validation set
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]

num_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

x_t = data[:-num_validation_samples]
y_t = labels[:-num_validation_samples]
x_val = data[-num_validation_samples:]
y_val = labels[-num_validation_samples:]


num_test_samples = int(TEST_SPLIT * x_t.shape[0])

x_train = x_t[:-num_test_samples]
y_train = y_t[:-num_test_samples]
x_test = x_t[-num_test_samples:]
y_test = y_t[-num_test_samples:]



print("Train: {} Val: {} Test:{}".format(x_train.shape[0],x_val.shape[0],x_test.shape[0]))

print('Preparing embedding matrix.')

# prepare embedding matrix
num_words = min(MAX_NUM_WORDS, len(word_index) + 1)
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
for word, i in word_index.items():
    if i >= MAX_NUM_WORDS:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

# load pre-trained word embeddings into an Embedding layer
# note that we set trainable = False so as to keep the embeddings fixed
embedding_layer = Embedding(num_words,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)

print('Training model.')


main_input = Input(shape=(MAX_SEQUENCE_LENGTH,),dtype='int32',name='main_input')
x = embedding_layer(main_input)
x = CuDNNLSTM(LSTM_HIDDEN_SIZE,return_sequences=True)(x)
Avg = keras.layers.core.Lambda(lambda x: K.mean(x, axis=1), output_shape=(LSTM_HIDDEN_SIZE, ))
x = Avg(x)
x = Dense(20)(x)
main_output = Activation('softmax')(x)

optimizer = Adam(lr=LEARNING_RATE,clipvalue=0.25)
m = Model(input=main_input, output=main_output)
m.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc'])


_, tmpfn = tempfile.mkstemp()
# Save the best model during validation and bail out of training early if we're not improving
callbacks = [ModelCheckpoint(tmpfn,monitor='val_acc', save_best_only=True, save_weights_only=True)]


history = m.fit(
        x_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=NUM_EPOCHS,
        validation_data=(x_val, y_val),
        callbacks=callbacks
     )


# Restore the best found model during validation
m.load_weights(tmpfn)

loss, acc = m.evaluate(x_test, y_test, batch_size=BATCH_SIZE)
print('Test loss / test accuracy = {:.4f} / {:.4f}'.format(loss, acc))


preds = m.predict(x_test, batch_size=BATCH_SIZE, verbose=0, steps=None)

pickler(OUTPUT_PATH,"predictions.pkl",preds)
pickler(OUTPUT_PATH,"x_test.pkl",x_test)
pickler(OUTPUT_PATH,"y_test.pkl",y_test)
pickler(OUTPUT_PATH,"history.pkl",history.history)
