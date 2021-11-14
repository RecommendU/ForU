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
        # pass
        self.train = pd.read_json('/home/ubuntu/workspace/jh/project/data/train.json')
        
        self.wv = w2v_preprocessing()
        self.tag_to_id, self.ply_tag = self.wv.tag_sprase_matrix()
    

        # models
        self.s2v_model = sent2vec.Sent2vecModel()
        self.s2v_model.load_model('/home/ubuntu/workspace/jh/project/model/s2v_model.bin')

        self.w2v_model = Word2Vec.load('model/w2v.model')

        self.p_title = pickle.load(open('/home/ubuntu/workspace/jh/project/model/p_title.pickle', 'rb'))
        self.p_tags = pickle.load(open('/home/ubuntu/workspace/jh/project/model/p_tags.pickle', 'rb'))
        self.p_new_tags = pickle.load(open('/home/ubuntu/workspace/jh/project/model/p_new_tags.pickle', 'rb'))
    
    def s2v_visual(self, text, k_n = 10):
        emb = s2v_model.embed_sentence(text)

        title_labels, title_distances = p_title.knn_query(emb, k = k_n, num_threads=8)
        tag_labels, tags_distances = p_tags.knn_query(emb, k = k_n, num_threads=8)
        new_tag_labels, new_tags_distances = p_new_tags.knn_query(emb, k = k_n, num_threads=8)

        reco_idx = list(chain.from_iterable(zip(title_labels.reshape(-1),tag_labels.reshape(-1),new_tag_labels.reshape(-1))))

        count = Counter(np.concatenate(train.iloc[reco_idx]['tags'].tolist()))
        words = dict(count.most_common())

        
        return words

    def w2v_visual(text):
        text = kp.konlpy_preprocessing(text).split()
        sims = w2v_model.wv.most_similar(text,topn=3)
        
        tags = []
        for value,sim in sims:
            tags.append(value)
        
        input_tag_id = [tag_to_id.get(x) for x in tags+text if tag_to_id.get(x) is not None]
        
        selected_playlists = []
        for tag_id in input_tag_id:
            # tag_id인 열 가져오기
            temp_list = ply_tag[:,tag_id].toarray().reshape(-1)
            # 그 중 1인 것만 playlist에 넣기
            selected_playlists.append(np.argwhere(temp_list == 1).reshape(-1))
            
        reco_idx = np.unique(np.concatenate(selected_playlists))

        count = Counter(np.concatenate(train.iloc[reco_idx,:].sort_values('like_cnt',ascending=False)[:30]['tags'].tolist()))
        words = dict(count.most_common())

        return words

    def wordcloud(text, ww = 0, save = 'off'):
        w = w2v_visual(text)
        k = s2v_visual(text,k_n=30)
        
        if ww == 1:
            fig, ax = plt.subplots()
            wc = WordCloud(font_path=r'/home/ubuntu/workspace/font/malgun.ttf',
                            max_words=500,
                            background_color='white',
                            colormap='prism',
                            width = 3000,
                            height = 2000)

            w = wc.generate_from_frequencies(w)
            ax.imshow(w, interpolation="bilinear")
            ax.set_title('Word2Vec Wordcloud')
            ax.axes.xaxis.set_visible(False)
            ax.axes.yaxis.set_visible(False)
            if save == 'on':
                plt.savefig(f'/home/ubuntu/workspace/wordcloud_images/w2v_{text}.png')

        elif ww == 2:
            fig, ax = plt.subplots()
            wc = WordCloud(font_path=r'/home/ubuntu/workspace/font/malgun.ttf',
                            max_words=500,
                            background_color='white',
                            colormap='prism',
                            width = 3000,
                            height = 2000)

            w = wc.generate_from_frequencies(s)
            ax.imshow(w, interpolation="bilinear")
            ax.set_title('Word2Vec Wordcloud')
            ax.axes.xaxis.set_visible(False)
            ax.axes.yaxis.set_visible(False)
            if save == 'on':
                plt.savefig(f'/home/ubuntu/workspace/wordcloud_images/w2v_{text}.png')
        
        else:
            fig, ax = plt.subplots(1,2)
            wc = WordCloud(font_path=r'/home/ubuntu/workspace/font/malgun.ttf',
                            max_words=500,
                            background_color='white',
                            colormap='prism',
                            width = 3000,
                            height = 2000)

            w = wc.generate_from_frequencies(w)
            ax[0].imshow(w, interpolation="bilinear")
            ax[0].set_title('Word2Vec Wordcloud')
            ax[0].axes.xaxis.set_visible(False)
            ax[0].axes.yaxis.set_visible(False)

            s = wc.generate_from_frequencies(k)
            ax[1].imshow(s, interpolation="bilinear")
            ax[1].set_title('Sent2Vec Wordcloud')
            ax[1].axes.xaxis.set_visible(False)
            ax[1].axes.yaxis.set_visible(False)
            if save == 'on':
                plt.savefig(f'/home/ubuntu/workspace/wordcloud_images/w2v_{text}.png')
            
        plt.show()
