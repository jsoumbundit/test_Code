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

def corpus1():
    model = Word2Vec.load('http://www.arts.chula.ac.th/ling/wp-content/uploads/TNCc5model.bin')
    #print(model.__dict__)
    #print(len(model.wv))
    #print(model.wv['นักเรียน'])
    #print(model.wv.most_similar('แมว'))
    inp = "corpus.th.text"
    outp1 = "corpus.th.model"
    model.save(outp1)
    print('Process saved file: corpus.th.model ready')



if __name__ == "__main__":
    corpus1()

