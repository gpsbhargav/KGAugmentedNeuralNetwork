from __future__ import print_function

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.set_random_seed(1234)
session = tf.Session(config=config)

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
from keras.layers import Dense, Input, Embedding, Activation, Softmax, dot
from keras.layers import CuDNNLSTM
from keras.models import Model,Sequential
from keras.layers import Conv3D,ZeroPadding2D,MaxPooling3D
from keras.layers import RepeatVector, Permute, Add, Concatenate, Reshape, Dot, Flatten
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint

np.random.seed(1234)

BASE_DIR = '.'
GLOVE_DIR = os.path.join('/home/bhargav/glove6B', 'glove')
#TEXT_DATA_DIR = os.path.join(BASE_DIR, '20_newsgroup')
DATA_DIR = "./data/Es_Rc"
DOC_PKL = "document_list.pick"
TARGET_PKL = "target_list.pick"
DOC_ENTS = "entity_vec_list.pick"

OUTPUT_PATH = './model_outputs/lstm_Es_Rc/'

MAX_SEQUENCE_LENGTH = 300
MAX_NUM_WORDS = 20000
EMBEDDING_DIM = 300
LSTM_HIDDEN_SIZE = 200
VALIDATION_SPLIT = 0.1
TEST_SPLIT = 0.2
NUM_EPOCHS = 30
BATCH_SIZE = 32
LEARNING_RATE = 0.001


KG_EMBEDDING_DIM = 50
RELATION_CLUSTER_FILE = './kg_clusters/Relation_cluster_vectors.txt'
#ENTITY_CLUSTER_FILE = './kg_clusters/entity_clusters.txt'
NUM_RELATIONS_PER_CLUSTER = 67
#NUM_ENTITIES_PER_CLUSTER = 342
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
#entity_clusters_np = get_clusters(ENTITY_CLUSTER_FILE,NUM_ENTITIES_PER_CLUSTER)

print("Cluster shapes: relations={}".format(relation_clusters_np.shape))



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

raw_docs = unpickler(DATA_DIR,DOC_PKL)
labels = unpickler(DATA_DIR,TARGET_PKL)
doc_ents = unpickler(DATA_DIR,DOC_ENTS)
doc_ents = np.array(doc_ents)

for doc in raw_docs:
    texts.append(" ".join(doc.split()[:MAX_SEQUENCE_LENGTH]))

repeated_relation_clusters = []    
for i in range(len(texts)):
    repeated_relation_clusters.append(relation_clusters_np)

relation_clusters_np = np.array(repeated_relation_clusters)

print("relations cluster shape after repeating={}".format(relation_clusters_np.shape))
    
                

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
doc_ents = doc_ents[indices]
relation_clusters_np = relation_clusters_np[indices]

num_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

x_t = data[:-num_validation_samples]
y_t = labels[:-num_validation_samples]
x_val = data[-num_validation_samples:]
y_val = labels[-num_validation_samples:]
relation_clusters_np_t = relation_clusters_np[:-num_validation_samples]
relation_clusters_np_val = relation_clusters_np[-num_validation_samples:]

ents_t = doc_ents[:-num_validation_samples]
ents_val = doc_ents[-num_validation_samples:]

num_test_samples = int(TEST_SPLIT * x_t.shape[0])

x_train = x_t[:-num_test_samples]
y_train = y_t[:-num_test_samples]
x_test = x_t[-num_test_samples:]
y_test = y_t[-num_test_samples:]
relation_clusters_np_train = relation_clusters_np_t[:-num_test_samples]
relation_clusters_np_test = relation_clusters_np_t[-num_test_samples:]

ents_train = ents_t[:-num_test_samples]
ents_test = ents_t[-num_test_samples:]



print('x_train:{} y_train:{} ents_train:{} x_val:{} y_val:{} ents_val:{}'.format(x_train.shape,y_train.shape,ents_train.shape, x_val.shape, y_val.shape, ents_val.shape))
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

#relation_clusters = K.variable(relation_clusters_np)
#entity_clusters = K.variable(entity_clusters_np)

Avg = keras.layers.core.Lambda(lambda x: K.mean(x, axis=1))#, output_shape=(KG_EMBEDDING_DIM, ))
DotProduct = keras.layers.core.Lambda(lambda x: K.dot(x[0],x[1]))#, output_shape=(KG_EMBEDDING_DIM, ))
Sum = keras.layers.core.Lambda(lambda x: K.sum(x,axis=1))#, output_shape=(KG_EMBEDDING_DIM, ))
RemoveLastCol = keras.layers.core.Lambda(lambda x: K.sum(x,axis=-1))
Transpose = keras.layers.core.Lambda(lambda x: K.transpose(x))
#FakeEClusterIn = keras.layers.core.Lambda(lambda x: entity_clusters)
FakeRClusterIn = keras.layers.core.Lambda(lambda x: relation_clusters)
RemoveLastDim = keras.layers.core.Lambda(lambda x: keras.backend.squeeze(x, -1))


main_input = Input(shape=(MAX_SEQUENCE_LENGTH,),dtype='int32',name='main_input')

x = embedding_layer(main_input)
x = CuDNNLSTM(KG_EMBEDDING_DIM,return_sequences=True)(x)
x = Avg(x)
x = Dense(KG_EMBEDDING_DIM)(x)
x = Activation('relu')(x)
relation_extraction = x

x = embedding_layer(main_input)
x = CuDNNLSTM(KG_EMBEDDING_DIM,return_sequences=True)(x)
x = Avg(x)
x = Dense(KG_EMBEDDING_DIM)(x)
x = Activation('relu')(x)
entity_extraction = x

x = embedding_layer(main_input)
x = CuDNNLSTM(LSTM_HIDDEN_SIZE,return_sequences=True)(x)
Avg = keras.layers.core.Lambda(lambda x: K.mean(x, axis=1), output_shape=(LSTM_HIDDEN_SIZE, ))
x = Avg(x)
x = Dense(LSTM_HIDDEN_SIZE)(x)
main_lstm_out = Activation('relu')(x)


#get representation for entity clusters
#x = FakeEClusterIn(x)
#x = Conv2D(filters=1,kernel_size=(5,1),strides=(5,1),input_shape=(342, 50, 1))(x)
#x = MaxPooling2D(pool_size=(3,1), strides=(3,1))(x)
#x = Conv2D(filters=1,kernel_size=(5,1),strides=(5,1))(x)
#x = MaxPooling2D(pool_size=(4, 1), strides=(1,1))(x)
#x = RemoveLastCol(x)
#x = Reshape([KG_EMBEDDING_DIM])(x)
#entity_cluster_reps = x
ents = Input(shape=(NUM_CLUSTERS,KG_EMBEDDING_DIM), name='ents', dtype='float32')#batch_shape=(BATCH_SIZE,NUM_CLUSTERS,KG_EMBEDDING_DIM),
print("ents",K.int_shape(ents))

#get representation for relationship clusters
relation_clusters = Input(shape=(NUM_CLUSTERS,NUM_RELATIONS_PER_CLUSTER,KG_EMBEDDING_DIM,1), name='relation_clusters', dtype='float32')
x = Conv3D(filters=1,kernel_size=(1,5,1),strides=(1,3,1),input_shape=(20,67, 50, 1))(relation_clusters)
x = MaxPooling3D(pool_size=(1,3, 1), strides=(1,2,1))(x)
x = Conv3D(filters=1,kernel_size=(1,3,1),strides=(1,2,1))(x)
x = MaxPooling3D(pool_size=(1,4, 1), strides=(1,1,1))(x)
print("before removing last col in cnn",K.int_shape(x))
x = RemoveLastCol(x)
print("after removing last col in cnn",K.int_shape(x))
x = Reshape([NUM_CLUSTERS,KG_EMBEDDING_DIM])(x)
print("after reshaping",K.int_shape(x))
relation_cluster_reps = x
print("relation_cluster_reps",K.int_shape(relation_cluster_reps))

#attention over relations
att_scores = dot([relation_cluster_reps,relation_extraction],axes=-1)
print("att_scores_relations",K.int_shape(att_scores))
#att_normalized = Activation('softmax',name="relation_attention")(att_scores)
att_normalized = Softmax(axis=-1,name='relation_attention')(att_scores)
relation_cluster_reps_T = Permute((2,1))(relation_cluster_reps)
print("relation_cluster_reps_T",K.int_shape(relation_cluster_reps_T))
the_relation = dot([relation_cluster_reps_T,att_normalized],axes=-1)
print("the_relation",K.int_shape(the_relation))

#attention over entities
att_scores = dot([ents, entity_extraction],axes=-1)
print("att_scores_entities",K.int_shape(att_scores))
#att_normalized = Activation('softmax',name="entity_attention")(att_scores)
att_normalized = Softmax(axis=-1,name='entity_attention')(att_scores)
ents_T = Permute((2,1))(ents)
print("relation_cluster_reps_T",K.int_shape(ents_T))
the_entity = dot([ents_T,att_normalized],axes=-1)
print("the_entity",K.int_shape(the_entity))

the_other_entity = Add()([the_relation,the_entity])
print("the_other_entity",K.int_shape(the_other_entity))




the_fact = Concatenate(axis=1)([the_entity, the_relation, the_other_entity])
print("the_fact",K.int_shape(the_fact))


lstm_hidden_and_fact = Concatenate(axis=1)([main_lstm_out, the_fact])
print("lstm_hidden_and_fact",K.int_shape(lstm_hidden_and_fact))


final_output = Dense(units=20, activation='softmax',name='final_output')(lstm_hidden_and_fact)


optimizer = Adam(lr=LEARNING_RATE,clipvalue=0.25)

m = Model(inputs=[main_input,ents,relation_clusters], outputs=[final_output])

m.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc'])

print("Model has {} parameters".format(m.count_params()))

def train_data_generator(docs,targets,doc_ents,relation_clusters,batch_size=BATCH_SIZE):
    while True:
        indices = np.arange(docs.shape[0])
        np.random.shuffle(indices)
        docs = docs[indices]
        targets = targets[indices]
        doc_ents = doc_ents[indices]
        for i in range(0,len(docs)-batch_size,batch_size):
            print("main_input:{} ents:{} relation_clusters:{}".format(docs[i:i+batch_size].shape,doc_ents[i:i+batch_size].shape,relation_clusters.shape ))
            yield ({'main_input':docs[i:i+batch_size], 'ents':doc_ents[i:i+batch_size] , 'relation_clusters':relation_clusters},
                   {'final_output':targets[i:i+batch_size]})
            
def val_data_generator(docs,targets,doc_ents,relation_clusters,batch_size=BATCH_SIZE):
    while True:
        indices = np.arange(docs.shape[0])
        np.random.shuffle(indices)
        docs = docs[indices]
        targets = targets[indices]
        doc_ents = doc_ents[indices]
        for i in range(0,len(docs)-batch_size,batch_size):
            yield ({'main_input':docs[i:i+batch_size], 'ents':doc_ents[i:i+batch_size], 'relation_clusters':relation_clusters},
                   {'final_output':targets[i:i+batch_size]})            


#m.fit_generator( generator=train_data_generator(x_train,y_train,ents_train,relation_clusters_np,batch_size=BATCH_SIZE), steps_per_epoch=x_train.shape[0]/BATCH_SIZE, 
#     epochs=NUM_EPOCHS,
#      validation_data=val_data_generator(x_val,y_val,ents_val,relation_clusters_np,batch_size=BATCH_SIZE), 
#      validation_steps=x_val.shape[0]/BATCH_SIZE
#     )


_, tmpfn = tempfile.mkstemp()
# Save the best model during validation and bail out of training early if we're not improving
callbacks = [ModelCheckpoint(tmpfn,monitor='val_acc', save_best_only=True, save_weights_only=True)]

history = m.fit(x={'main_input':x_train, 'ents':ents_train, 'relation_clusters':relation_clusters_np_train},
      y={'final_output':y_train},
      batch_size=BATCH_SIZE, epochs=NUM_EPOCHS, verbose=1, callbacks=callbacks, validation_split=0.0, 
      validation_data=({'main_input':x_val, 'ents':ents_val, 'relation_clusters':relation_clusters_np_val},{'final_output':y_val}) , 
      shuffle=True, class_weight=None, sample_weight=None, 
      initial_epoch=0, steps_per_epoch=None, validation_steps=None)


# Restore the best found model during validation
m.load_weights(tmpfn)

loss, acc = m.evaluate({'main_input':x_test, 'ents':ents_test, 'relation_clusters':relation_clusters_np_test}, y_test, batch_size=BATCH_SIZE)
print('Test loss / test accuracy = {:.4f} / {:.4f}'.format(loss, acc))


preds = m.predict({'main_input':x_test, 'ents':ents_test, 'relation_clusters':relation_clusters_np_test}, batch_size=BATCH_SIZE, verbose=0, steps=None)

entity_attention_model = Model(inputs=m.input,outputs=m.get_layer("entity_attention").output)
relation_attention_model = Model(inputs=m.input,outputs=m.get_layer("relation_attention").output)

e_attention_test = entity_attention_model.predict({'main_input':x_test, 'ents':ents_test, 'relation_clusters':relation_clusters_np_test}, batch_size=BATCH_SIZE, verbose=0, steps=None)

r_attention_test = relation_attention_model.predict({'main_input':x_test, 'ents':ents_test, 'relation_clusters':relation_clusters_np_test}, batch_size=BATCH_SIZE, verbose=0, steps=None)


#e_attention_train = entity_attention_model.predict({'main_input':x_train, 'ents':ents_train, 'relation_clusters':relation_clusters_np_train}, batch_size=BATCH_SIZE, verbose=0, steps=None)

#r_attention_train = relation_attention_model.predict({'main_input':x_train, 'ents':ents_train, 'relation_clusters':relation_clusters_np_train}, batch_size=BATCH_SIZE, verbose=0, steps=None)

pickler(OUTPUT_PATH,"history.pkl",history.history)
pickler(OUTPUT_PATH,"predictions.pkl",preds)
pickler(OUTPUT_PATH,"x_test.pkl",x_test)
pickler(OUTPUT_PATH,"y_test.pkl",y_test)
pickler(OUTPUT_PATH,"e_attention_test.pkl",e_attention_test)
pickler(OUTPUT_PATH,"r_attention_test.pkl",r_attention_test)
pickler(OUTPUT_PATH,"ents_test.pkl",ents_test)


