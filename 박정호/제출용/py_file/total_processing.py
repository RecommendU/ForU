# word2vec 전처리

import numpy as np
import pandas as pd
from itertools import chain
from collections import Counter
from scipy.sparse import *
from scipy.sparse.linalg import svds
from collections import Counter,defaultdict
from itertools import chain, combinations
from gensim import corpora

# 한글 전처리

from konlpy.tag import Okt, Mecab
import nltk
from nltk.corpus import stopwords
import pickle

class konlpy_preprocessing:
    def __init__(self):
        self.m = Mecab('/home/ubuntu/workspace/mecab-ko-dic-2.1.1-20180720')
        self.stopword = pickle.load(open('/home/ubuntu/workspace/jh/project/data/10_28_stopword.pickle','rb'))


    def konlpy_preprocessing(self, text, removes_stopwords = True):
        
        parts_of_speech = []    
        # mecab으로 원하는 품사만 뽑아오기
        for word, tag in self.m.pos(text):
            if tag in ['NNP', 'NNG', 'VA', 'VV', 'SL', 'SN', 'XR', 'VA+ETM', 'VV+EC+VX+ETM']:
                parts_of_speech.append(word.lower())
        
        #  stopwords에 있는 단어 제거

        parts_of_speech = [token for token in parts_of_speech if not token in self.stopword]    

        stops = set(stopwords.words('english')) # 영어 불용어 불러오기

        parts_of_speech = [w for w in parts_of_speech if not w in stops]
        
        return " ".join(parts_of_speech)