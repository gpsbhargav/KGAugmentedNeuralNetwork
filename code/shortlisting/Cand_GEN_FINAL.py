#Read All Entities name
import codecs
import string
f=codecs.open("ranked_entities_names.txt",encoding = 'utf8')
lines=f.readlines()
names=[]
mid = []
replace_punctuation = str.maketrans(string.punctuation, ' '*len(string.punctuation))
for x in lines:
    st = x.split('\t')[1][:-1].lower()
    l = st.translate(replace_punctuation)
    names.append(l)
    mid.append(x.split('\t')[0])
f.close()

f=codecs.open("entity2id.txt",encoding = 'utf8')
lines=f.readlines()
entity2id = {}
id2entity = {}
for x in lines:
    entity = x.split('\t')[0]
    num = (int)(x.split('\t')[1][:-2])
    entity2id[entity] = num
    id2entity[num] = entity
    
f.close()

#Assuming entities are arranged according to their ranks
from collections import Counter
import editdistance
def candidate_gen(entities,grams):
    entity_count = Counter(entities)
    #An entity whose label exactly matches a n-gram is added to the set of candidates Cs
    C = []
    for grm in grams:
        if entity_count[grm] >= 1:
            C.append(grm)
    C = list(set(C))
               
     # An n-gram that is fully contained inside a bigger n-gram that matched at least one entity label is dis-
     # carded, unless the bigger n-gram starts with one ofthe following words: “the”, “a”, “an”, “of ”, “on”, “at”,
     #  “by” 
    first_words = ["the", "a", "an", "of", "on", "at","by" ]
     
    for grm  in grams:
        if grm in C:
            grams.remove(grm)
        else:
            for word in C:
                if grm in get_ngram(word.split()):
                    #print(grm,word)
                    first = word.split()[0]
                    if first not in first_words:
                        grams.remove(grm)
                        break
    C = list(set(C))
    
    if len(C)>=20:
        return C
    
    #In case an exact match is not found for an n-gram we search for entities whose label has an edit distance
    #with the n-gram of at most 1 and add them to Cs.
    for i,grm in enumerate(grams):
        l = [word for word in entity_count.keys() if editdistance.eval(grm, word)<=1]
        if len(l)>5:
            C = C + l[:5]
        else:
            C = C + l
        if(len(set(C))) >=20:
            return C
    
    return C


from nltk import ngrams
def get_ngram(word_list):
    gram = []
    length = 3
    if len(word_list)<3:
        length = len(word_list)
    for i in range(length,0,-1):
        igram = ngrams(word_list, i)
        for grm in igram:
            gram.append(' '.join(grm))
    return gram

print("loading")
from sklearn.datasets import fetch_20newsgroups
newsgroups_train = fetch_20newsgroups(subset='all',remove=('headers', 'footers', 'quotes'))
print("completed loading")
def compare(item1):
    return len(item1.split())


import nltk.data
import codecs
import string
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize,word_tokenize
stop_words = set(stopwords.words('english'))
from nltk import ngrams

vectors = {}
vectorsid = {}
for i in range(len(newsgroups_train.data)):
    data = newsgroups_train.data[i].split("\n")
    sen = []
    for line in data:
        sen = sen + sent_tokenize(line.lower())
    replace_punctuation = str.maketrans(string.punctuation, ' '*len(string.punctuation))
    for j,s in enumerate(sen):
        sen[j] = s.translate(replace_punctuation)
    
    gram = []
    for s in sen:
        tokens = [w for w in word_tokenize(s) if not w in stop_words]
        gram = gram + get_ngram(word_tokenize(s))
    
    #print(len(gram))
    C = candidate_gen(names,gram)
    C = set(C)
    C = sorted(C, key=compare,reverse=True)
    
    mids =[]
    ids = []
    for nam in set(C):
        mids.append(mid[names.index(nam)])
        ids.append(entity2id[mid[names.index(nam)]])
    
    if len(ids)>20:
        ids = ids[:20]
    
    vectors[i] = mids
    vectorsid[i] = ids
    print(i)

import pickle
pickle_out = open("vectorsid.pick","wb")
pickle.dump(vectorsid, pickle_out)
pickle_out.close()

cnt = 0
for i in vectorsid:
    if len(vectorsid[i])!=20:
        print(i,len(vectorsid[i]),cnt)
        cnt = cnt+1
        
print("pickling Now")
import numpy as np
entity2vec = np.loadtxt('entity2vec.bern')

list_of_entity_vecs = []
for i in vectorsid:
    if len(vectorsid[i]) == 20:
        list_of_entity_vecs.append(entity2vec[vectorsid[i]])
    elif len(vectorsid[i]) > 0:
        l = vectorsid[i] + list(np.random.choice(vectorsid[i],20-len(vectorsid[i])))
        list_of_entity_vecs.append(entity2vec[l])
    else:
        list_of_entity_vecs.append(np.zeros((20,50)))#dim of each vector is 50
        
doc_list = list(newsgroups_train.data)
pickle_out = open("document_list.pick","wb")
pickle.dump(doc_list, pickle_out)
pickle_out.close()

target_list = list(newsgroups_train.target)
pickle_out = open("target_list.pick","wb")
pickle.dump(target_list, pickle_out)
pickle_out.close()

pickle_out = open("entity_vec_list.pick","wb")
pickle.dump(list_of_entity_vecs, pickle_out)
pickle_out.close()
