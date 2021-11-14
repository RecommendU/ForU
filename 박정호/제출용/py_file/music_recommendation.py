import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['font.size'] = 15
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.figsize'] = 16,8

from collections import Counter,defaultdict
from itertools import chain, combinations

import warnings
warnings.filterwarnings('ignore')

import imshow
from wordcloud import WordCloud
from PIL import Image

import re
from collections import Counter
from itertools import chain
import pickle
import math

import sent2vec
import hnswlib
from gensim.models import Word2Vec
from gensim import corpora
from scipy.sparse import *
from scipy.sparse.linalg import svds

# -------
from py_file.total_processing import *


class music_recommendation:
    def __init__(self):
        self.kp = konlpy_preprocessing()
        self.song_meta = pd.read_json('/home/ubuntu/workspace/jh/project/data/song_meta.json')
        self.song_meta['artists'] = self.song_meta['artist_name_basket'].apply(lambda x : " ".join(x))

        self.train = pd.read_json('/home/ubuntu/workspace/jh/project/data/train.json')
    
        # models
        self.s2v_model = sent2vec.Sent2vecModel()
        self.s2v_model.load_model('/home/ubuntu/workspace/jh/project/model/s2v_model.bin')
        self.p_title_s2v = pickle.load(open('/home/ubuntu/workspace/jh/project/model/p_title.pickle', 'rb'))
        self.p_tags_s2v = pickle.load(open('/home/ubuntu/workspace/jh/project/model/p_tags.pickle', 'rb'))
        self.p_new_tags_s2v = pickle.load(open('/home/ubuntu/workspace/jh/project/model/p_new_tags.pickle', 'rb'))
        
        self.w2v_model = Word2Vec.load('/home/ubuntu/workspace/jh/project/model/w2v.model')
        self.p_tags_w2v = pickle.load(open('/home/ubuntu/workspace/jh/project/model/w2v_p_tags.pickle', 'rb'))
        self.p_titles_w2v = pickle.load(open('/home/ubuntu/workspace/jh/project/model/w2v_p_titles.pickle', 'rb'))
        self.p_new_tags_w2v = pickle.load(open('/home/ubuntu/workspace/jh/project/model/w2v_p_new_tags.pickle', 'rb'))


    def s2v_recommendation(self, text, k_n = 2):
        test_tag = self.kp.konlpy_preprocessing(text)

        test_tag = test_tag.split()
        
        test_tag = " ".join([i for i in test_tag if self.s2v_model.embed_sentence(i).sum() != 0])

        if self.s2v_model.embed_sentence(test_tag).sum() == 0.0:
            print('검색 결과를 찾을 수 없습니다!')

        else:
            emb = self.s2v_model.embed_sentence(test_tag)
            title_labels, title_distances = self.p_title_s2v.knn_query(emb, k = k_n, num_threads=8)
            tag_labels, tags_distances = self.p_tags_s2v.knn_query(emb, k = k_n, num_threads=8)
            new_tag_labels, new_tags_distances = self.p_new_tags_s2v.knn_query(emb, k = k_n, num_threads=8)

            reco_idx = list(chain.from_iterable(zip(title_labels.reshape(-1),tag_labels.reshape(-1),new_tag_labels.reshape(-1))))
            
            song_idx = list(set(np.concatenate(self.train.iloc[reco_idx]['songs'].tolist())))

            reco_songs = self.song_meta.iloc[song_idx,:][['song_name','artists']]
            
            return reco_songs.sample(30)

    def w2v_recommendation(self, text, k_n=2):
        test_tag = self.kp.konlpy_preprocessing(text)

        test_tag = test_tag.split()

        tag_emb = []
        try:
            for txt in test_tag:
                tag_emb.append(self.w2v_model.wv.get_vector(txt))
        except ZeroDivisionError and KeyError:
            tag_emb.append(np.zeros(100))

        emb = sum(tag_emb)/len(tag_emb)

        if sum(emb) == 0:
            print('다시 입력해주세요!')
        else:
            title_labels, title_distances = self.p_titles_w2v.knn_query(emb, k = k_n, num_threads=8)
            tag_labels, tags_distances = self.p_tags_w2v.knn_query(emb, k = k_n, num_threads=8)
            new_tag_labels, new_tags_distances = self.p_new_tags_w2v.knn_query(emb, k = k_n, num_threads=8)

            reco_idx = list(chain.from_iterable(zip(title_labels.reshape(-1),tag_labels.reshape(-1),new_tag_labels.reshape(-1))))

            song_idx = list(set(np.concatenate(self.train.iloc[reco_idx]['songs'].tolist())))

            reco_songs = self.song_meta.iloc[song_idx,:][['song_name','artists']]

            return reco_songs.sample(30)

    def word_count(self, text,k_n = 10):
        test_tag = self.kp.konlpy_preprocessing(text)

        test_tag = test_tag.split()
        test_tag = " ".join([i for i in test_tag if self.s2v_model.embed_sentence(i).sum() != 0])

        try:
            text = test_tag.split()
            tag_emb = [self.w2v_model.wv.get_vector(x) for x in text]

            emb = sum(tag_emb)/len(tag_emb)

            title_labels, title_distances = self.p_titles_w2v.knn_query(emb, k = k_n, num_threads=8)
            tag_labels, tags_distances = self.p_tags_w2v.knn_query(emb, k = k_n, num_threads=8)
            new_tag_labels, new_tags_distances = self.p_new_tags_w2v.knn_query(emb, k = k_n, num_threads=8)
        
            reco_idx = list(chain.from_iterable(zip(title_labels.reshape(-1),tag_labels.reshape(-1),new_tag_labels.reshape(-1))))
            w2v_count = Counter(np.concatenate(self.train.iloc[reco_idx]['tags'].tolist()))

            w2v_words = dict(w2v_count.most_common())
        
        except ValueError and ZeroDivisionError:
            
            print('다시 입력해주세요!')
            return None,None



        if self.s2v_model.embed_sentence(test_tag).sum() == 0.0:
            print()    
        else:
            emb = self.s2v_model.embed_sentence(test_tag)
            title_labels, title_distances = self.p_title_s2v.knn_query(emb, k = k_n, num_threads=8)
            tag_labels, tags_distances = self.p_tags_s2v.knn_query(emb, k = k_n, num_threads=8)
            new_tag_labels, new_tags_distances = self.p_new_tags_s2v.knn_query(emb, k = k_n, num_threads=8)

            reco_idx = list(chain.from_iterable(zip(title_labels.reshape(-1),tag_labels.reshape(-1),new_tag_labels.reshape(-1))))

            s2v_count = Counter(np.concatenate(self.train.iloc[reco_idx]['tags'].tolist()))
            s2v_words = dict(s2v_count.most_common())

            return w2v_words,s2v_words


    def color_func(self,word, font_size, position,orientation,random_state=None, **kwargs):
        return(f"hsl({np.random.randint(18,45)},{np.random.randint(99,100)}%, {np.random.randint(30,70)}%)")
    
    
    def wc_compare(self, text, ww = 'compare', save = 'off',k_n=10):
        w2v_words, s2v_words = self.word_count(text)

        if ww == 'w':
            fig, ax = plt.subplots()
            wc = WordCloud(font_path=r'/home/ubuntu/workspace/font/GmarketSansTTFMedium.ttf',
                            max_words=100,
                            background_color='white',
                            color_func = self.color_func,
                            relative_scaling  = 0.2,
                            width = 3000,
                            height = 2000)

            w = wc.generate_from_frequencies(w2v_words)
            ax.imshow(w, interpolation="bilinear")
            ax.set_title('Word2Vec Wordcloud')
            ax.axes.xaxis.set_visible(False)
            ax.axes.yaxis.set_visible(False)
            if save == 'on':
                plt.savefig(f'/home/ubuntu/workspace/wordcloud_images/w2v_{text}.png')

        elif ww == 's':
            fig, ax = plt.subplots()
            wc = WordCloud(font_path=r'/home/ubuntu/workspace/font/GmarketSansTTFMedium.ttf',
                            max_words=100,
                            background_color='white',
                            color_func = self.color_func,
                            relative_scaling  = 0.2,
                            width = 3000,
                            height = 2000)

            s = wc.generate_from_frequencies(s2v_words)
            ax.imshow(s, interpolation="bilinear")
            ax.set_title('Sent2Vec Wordcloud')
            ax.axes.xaxis.set_visible(False)
            ax.axes.yaxis.set_visible(False)
            if save == 'on':
                plt.savefig(f'/home/ubuntu/workspace/wordcloud_images/s2v_{text}.png')
            
        
        else:
            fig, ax = plt.subplots(1,2)
            wc = WordCloud(font_path=r'/home/ubuntu/workspace/font/GmarketSansTTFMedium.ttf',
                            max_words=100,
                            background_color='white',
                            color_func = self.color_func,
                            relative_scaling  = 0.2,
                            width = 3000,
                            height = 2000)

            w = wc.generate_from_frequencies(w2v_words)
            ax[0].imshow(w, interpolation="bilinear")
            ax[0].set_title('Word2Vec Wordcloud')
            ax[0].axes.xaxis.set_visible(False)
            ax[0].axes.yaxis.set_visible(False)

            s = wc.generate_from_frequencies(s2v_words)
            ax[1].imshow(s, interpolation="bilinear")
            ax[1].set_title('Sent2Vec Wordcloud')
            ax[1].axes.xaxis.set_visible(False)
            ax[1].axes.yaxis.set_visible(False)
            
            if save == 'on':
                plt.savefig(f'/home/ubuntu/workspace/wordcloud_images/compare_{text}.png')
            
        plt.show()