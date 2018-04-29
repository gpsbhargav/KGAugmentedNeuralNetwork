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
import numpy as np
import dill as pickle
import tempfile

import keras
from keras.preprocessing.text import Tokenizer
import keras.backend as K
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Input, Embedding, Activation, Softmax
from keras.layers import CuDNNLSTM
from keras.models import Model,Sequential
from keras.layers import Conv2D,ZeroPadding2D,MaxPooling2D
from keras.layers import RepeatVector, Permute, Add, Concatenate, Reshape, Dot
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint

np.random.seed(100)

BASE_DIR = '.'
GLOVE_DIR = os.path.join('/home/bhargav/glove6B', 'glove')
#TEXT_DATA_DIR = os.path.join(BASE_DIR, '20_newsgroup')
DATA_DIR = "/home/bhargav/nlu_project/keras_n20/data/Es_Rc"
DOC_PKL = "document_list.pick"
TARGET_PKL = "target_list.pick"

OUTPUT_PATH = './model_outputs/lstm_and_kg/'

MAX_SEQUENCE_LENGTH = 300
MAX_NUM_WORDS = 20000
EMBEDDING_DIM = 300
LSTM_HIDDEN_SIZE = 200
VALIDATION_SPLIT = 0.1
TEST_SPLIT = 0.2
NUM_EPOCHS = 50
BATCH_SIZE = 256
LEARNING_RATE = 0.001


KG_EMBEDDING_DIM = 50
RELATION_CLUSTER_FILE = './kg_clusters/Relation_cluster_vectors.txt'
ENTITY_CLUSTER_FILE = './kg_clusters/entity_clusters.txt'
NUM_RELATIONS_PER_CLUSTER = 67
NUM_ENTITIES_PER_CLUSTER = 342
NUM_CLUSTERS = 20

def get_clusters(cluster_file, num_things_per_cluster):
    clusters = [] #np.ones(shape=(NUM_RELATIONS_PER_CLUSTER,KG_EMBEDDING_DIM))

    with open(cluster_file,'r',encoding='utf8') as f:
        lines = []
        for line in f:
            elements = line.split()
            x = [ [e] for e in elements ]
            lines.append(x)

    for i in range(0,len(lines)-num_things_per_cluster+1,num_things_per_cluster):
        #print("appending: {} to {}".format(i,i+num_things_per_cluster))
        clusters.append(lines[i:i+num_things_per_cluster])
    
    clusters = np.asarray(clusters, dtype='float32')
    return clusters

def pickler(path,pkl_name,obj):
    with open(os.path.join(path, pkl_name), 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def unpickler(path,pkl_name):
    with open(os.path.join(path, pkl_name) ,'rb') as f:
        obj = pickle.load(f)
    return obj


# num_clusters x num_vectors_per_cluster x dimension_of_vector
relation_clusters_np = get_clusters(RELATION_CLUSTER_FILE,NUM_RELATIONS_PER_CLUSTER)
entity_clusters_np = get_clusters(ENTITY_CLUSTER_FILE,NUM_ENTITIES_PER_CLUSTER)

print("Cluster shapes- entity:{} , relations={}".format(entity_clusters_np.shape,relation_clusters_np.shape))



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



print('x_train:{} y_train:{} x_val:{} y_val:{} '.format(x_train.shape,y_train.shape, x_val.shape, y_val.shape))
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

relation_clusters = K.variable(relation_clusters_np)
entity_clusters = K.variable(entity_clusters_np)

Avg = keras.layers.core.Lambda(lambda x: K.mean(x, axis=1))#, output_shape=(KG_EMBEDDING_DIM, ))
DotProduct = keras.layers.core.Lambda(lambda x: K.dot(x[0],x[1]))#, output_shape=(KG_EMBEDDING_DIM, ))
Sum = keras.layers.core.Lambda(lambda x: K.sum(x,axis=1))#, output_shape=(KG_EMBEDDING_DIM, ))
RemoveLastCol = keras.layers.core.Lambda(lambda x: K.sum(x,axis=-1))
Transpose = keras.layers.core.Lambda(lambda x: K.transpose(x))
FakeEClusterIn = keras.layers.core.Lambda(lambda x: entity_clusters)
FakeRClusterIn = keras.layers.core.Lambda(lambda x: relation_clusters)


main_input = Input(shape=(MAX_SEQUENCE_LENGTH,),dtype='int32',name='main_input')

x = embedding_layer(main_input)
x = CuDNNLSTM(KG_EMBEDDING_DIM,return_sequences=True)(x)
x = Avg(x)
x = Dense(KG_EMBEDDING_DIM)(x)
x = Activation('relu')(x)
#relation_extraction = Reshape([KG_EMBEDDING_DIM])(x)
relation_extraction = Transpose(x)

x = embedding_layer(main_input)
x = CuDNNLSTM(KG_EMBEDDING_DIM,return_sequences=True)(x)
x = Avg(x)
x = Dense(KG_EMBEDDING_DIM)(x)
x = Activation('relu')(x)
#entity_extraction = Reshape([KG_EMBEDDING_DIM])(x)
entity_extraction = Transpose(x)

x = embedding_layer(main_input)
x = CuDNNLSTM(LSTM_HIDDEN_SIZE,return_sequences=True)(x)
Avg = keras.layers.core.Lambda(lambda x: K.mean(x, axis=1), output_shape=(LSTM_HIDDEN_SIZE, ))
x = Avg(x)
x = Dense(LSTM_HIDDEN_SIZE)(x)
main_lstm_out = Activation('relu')(x)


#get representation for entity clusters
#e_clusters = Input(tensor=entity_clusters,name='e_clusters')
x = FakeEClusterIn(x)
x = Conv2D(filters=1,kernel_size=(5,1),strides=(5,1),input_shape=(342, 50, 1))(x)
x = MaxPooling2D(pool_size=(3,1), strides=(3,1))(x)
x = Conv2D(filters=1,kernel_size=(5,1),strides=(5,1))(x)
x = MaxPooling2D(pool_size=(4, 1), strides=(1,1))(x)
print("Before removing last col",K.int_shape(x))
x = RemoveLastCol(x)
print("after removing last col",K.int_shape(x))
entity_cluster_reps = Reshape([KG_EMBEDDING_DIM],name='entity_cluster_reps')(x)
print("entity_cluster_reps(after reshape)",K.int_shape(entity_cluster_reps))


#get representation for relationship clusters
#r_clusters = Input(tensor=relation_clusters,name='r_clusters')
x = FakeRClusterIn(x)
x = Conv2D(filters=1,kernel_size=(5,1),strides=(3,1),input_shape=(67, 50, 1))(x)
x = MaxPooling2D(pool_size=(3, 1), strides=(2,1))(x)
x = Conv2D(filters=1,kernel_size=(3,1),strides=(2,1))(x)
x = MaxPooling2D(pool_size=(4, 1), strides=(1,1))(x)
print("Before removing last col",K.int_shape(x))
x = RemoveLastCol(x)
print("after removing last col",K.int_shape(x))
relation_cluster_reps = Reshape([KG_EMBEDDING_DIM],name='relation_cluster_reps')(x)
print("relation_cluster_reps(after reshape)",K.int_shape(relation_cluster_reps))


#attention over relations
att_scores = DotProduct([relation_cluster_reps,relation_extraction])
print("att_scores_relations",K.int_shape(att_scores))
#att_normalized = Activation('softmax',name='relation_attention')(att_scores)
att_normalized = Softmax(axis=-1,name='relation_attention')(att_scores)
print("att_normalized",K.int_shape(att_normalized))
the_relation = DotProduct([Transpose(relation_cluster_reps),att_normalized])
print("the_relation",K.int_shape(the_relation))

#attention over entities
att_scores = DotProduct([entity_cluster_reps, entity_extraction])
print("att_scores_entities",K.int_shape(att_scores))
#att_normalized = Activation('softmax',name='entity_attention')(att_scores)
att_normalized = Softmax(axis=-1,name='entity_attention')(att_scores)
print("att_normalized",K.int_shape(att_normalized))
the_entity = DotProduct([Transpose(entity_cluster_reps),att_normalized])
print("the_entity",K.int_shape(the_entity))

the_other_entity = Add()([the_relation,the_entity])
print("the_other_entity",K.int_shape(the_other_entity))

the_fact = Concatenate(axis=0)([the_entity, the_relation, the_other_entity])
print("the_fact",K.int_shape(the_fact))

lstm_hidden_and_fact = Concatenate(axis=0)([Transpose(main_lstm_out), the_fact])
print("lstm_hidden_and_fact",K.int_shape(lstm_hidden_and_fact))
#input("continue?")

final_output = Dense(units=20, activation='softmax')(Transpose(lstm_hidden_and_fact))


optimizer = Adam(lr=LEARNING_RATE,clipvalue=0.25)

m = Model(inputs=[main_input], outputs=[final_output])

m.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc'])

_, tmpfn = tempfile.mkstemp()
# Save the best model during validation and bail out of training early if we're not improving
callbacks = [ModelCheckpoint(tmpfn,monitor='val_acc', save_best_only=True, save_weights_only=True)]

m.summary()

history = m.fit({'main_input':x_train, 'e_clusters':None, 'e_clusters':None}, y_train,
          batch_size=BATCH_SIZE,
          epochs=NUM_EPOCHS,
          validation_data=(x_val, y_val), callbacks=callbacks)

# Restore the best found model during validation
m.load_weights(tmpfn)

loss, acc = m.evaluate({'main_input':x_test, 'e_clusters':None, 'e_clusters':None}, y_test, batch_size=BATCH_SIZE)
print('Test loss / test accuracy = {:.4f} / {:.4f}'.format(loss, acc))


preds = m.predict({'main_input':x_test, 'e_clusters':None, 'e_clusters':None}, batch_size=BATCH_SIZE, verbose=0, steps=None)

#pickler(OUTPUT_PATH,"history.pkl",history.history)

#entity_attention_model = Model(inputs=m.input,outputs=m.get_layer("entity_attention").output)
#relation_attention_model = Model(inputs=m.input,outputs=m.get_layer("relation_attention").output)

#entity_cluster_reps_model = Model(inputs=m.input,outputs=m.get_layer("entity_cluster_reps").output)
#relation_cluster_reps_model = Model(inputs=m.input,outputs=m.get_layer("relation_cluster_reps").output)

#entity_attention_model.summary()
#input("continue?")

#e_attention = entity_attention_model.predict({'main_input':x_test, 'e_clusters':None, 'e_clusters':None}, batch_size=BATCH_SIZE, verbose=0, steps=None)

#r_attention = relation_attention_model.predict({'main_input':x_test, 'e_clusters':None, 'e_clusters':None}, batch_size=BATCH_SIZE, verbose=0, steps=None)

#e_cluster_reps = entity_cluster_reps_model.predict({'main_input':x_test, 'e_clusters':None, 'e_clusters':None}, batch_size=BATCH_SIZE, verbose=0, steps=None)

#r_cluster_reps = relation_cluster_reps_model.predict({'main_input':x_test, 'e_clusters':None, 'e_clusters':None}, batch_size=BATCH_SIZE, verbose=0, steps=None)

#pickler(OUTPUT_PATH,"predictions.pkl",preds)
#pickler(OUTPUT_PATH,"x_test.pkl",x_test)
#pickler(OUTPUT_PATH,"y_test.pkl",y_test)
#pickler(OUTPUT_PATH,"e_attention.pkl",e_attention)
#pickler(OUTPUT_PATH,"r_attention.pkl",r_attention)
#pickler(OUTPUT_PATH,"e_cluster_reps.pkl",e_cluster_reps)
#pickler(OUTPUT_PATH,"r_cluster_reps.pkl",r_cluster_reps)



