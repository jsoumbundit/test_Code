import pandas as pd
import numpy as np
import string
import logging
import os.path
import sys
import multiprocessing
from gensim.corpora import WikiCorpus
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

if __name__ == "__main__":

    #print(model.__dict__)
    #print(len(model.wv))
    #print(model.wv['นักเรียน'])
    #print(model.wv.most_similar('แมว'))
    #print(model.wv.key_to_index['นักเรียน'])

    outp1 = "corpus.th.model"
    model = Word2Vec.load(outp1)
    #print(len(model.wv))
    #print(model.wv.key_to_index['ว่างเปล่า'])
    #print(model.wv.key_to_index)
    #print(model.wv.key_to_index)
    #for k,v in model.wv.key_to_index.items():
    #    print(k,v)
    print('model.wv[''มาตรา'']')
    print(model.wv['มาตรา'])
    print(model.wv.key_to_index['มาตรา'])
