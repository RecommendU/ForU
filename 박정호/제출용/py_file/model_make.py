import numpy as np
import pandas as pd
import pickle
from itertools import chain
import sent2vec
import hnswlib
from gensim.models import Word2Vec
from gensim import corpora
import os

train = pd.read_json('data/1108_new_train.json')

tag_title = pd.read_json('/home/ubuntu/workspace/jh/project/data/1108_konlpy_text.json')

train['tag'] = tag_title['tags']
train['new_tags__'] = train['new_tags'].apply(lambda x : " ".join(x))
train['tags__'] = train['tag'].apply(lambda x : " ".join(x))

ply_title = train['plylst_title'].tolist()
tags = train['tags__'].tolist()
new_tags = train['new_tags__'].tolist()

s2v_text = train['new_tags__'].tolist()

f = open("./data/s2v_text.txt", 'w')
for i in s2v_text:
    x = f'{i} \n'
    f.write(x)

f.close

class model_make:
    def __init__(self):
        # self.train = pd.read_json('/home/ubuntu/workspace/jh/project/data/1108_new_train.json')
        pass

    def s2v_model_make(self, neg = 10):
        cmd = "/home/ubuntu/workspace/sent2vec/fasttext sent2vec" 
        cmd += " -input /home/ubuntu/workspace/jh/project/data/s2v_text.txt" 
        cmd += " -output ./model/s2v_model" 
        cmd += " -minCount 0" 
        cmd += " -dim 100" 
        cmd += " -epoch 20" 
        cmd += " -lr 0.2" 
        cmd += " -wordNgrams 3" 
        cmd += " -loss ns" 
        cmd += f" -neg {neg}" 
        cmd += " -thread 16" 
        cmd += " -t 0.000005" 
        cmd += " -dropoutK 4" 
        cmd += " -minCountLabel 20" 
        cmd += " -bucket 100000" 
        cmd += " -maxVocabSize 20000" 
        cmd += " -numCheckPoints 1"

        os.system(cmd)

        print(neg)

        s2v_model = sent2vec.Sent2vecModel()
        s2v_model.load_model('/home/ubuntu/workspace/jh/project/model/s2v_model.bin')

        # title
        title_emb = s2v_model.embed_sentences(ply_title)
        title_idx_to_emb = {}
        for idx, emb in enumerate(title_emb):
            if emb.sum() != 0:
                title_idx_to_emb[idx] = emb

        # tags
        tags_emb = s2v_model.embed_sentences(tags)
        tags_idx_to_emb = {}
        for idx, emb in enumerate(tags_emb):
            if emb.sum() != 0:
                tags_idx_to_emb[idx] = emb

        # new_tags
        new_tags_emb = s2v_model.embed_sentences(new_tags)
        new_tags_idx_to_emb = {}
        for idx, emb in enumerate(new_tags_emb):
            if emb.sum() != 0:
                new_tags_idx_to_emb[idx] = emb

        sp = 'cosine'
        dim = s2v_model.get_emb_size()
        
            
        # title
        p_title = hnswlib.Index(space = sp, dim = dim)
        p_title.init_index(max_elements = len(title_idx_to_emb), 
                        ef_construction = 100, M = 16, random_seed = 100)
        p_title.add_items(list(title_idx_to_emb.values()), list(title_idx_to_emb.keys()))

        # tags
        p_tags = hnswlib.Index(space = sp, dim = dim)
        p_tags.init_index(max_elements = len(tags_idx_to_emb), 
                        ef_construction = 100, M = 16, random_seed = 100)
        p_tags.add_items(list(tags_idx_to_emb.values()), list(tags_idx_to_emb.keys()))

        # new_tags
        p_new_tags = hnswlib.Index(space = sp, dim = dim)
        p_new_tags.init_index(max_elements = len(new_tags_idx_to_emb), 
                            ef_construction = 100, M = 16, random_seed = 100)
        p_new_tags.add_items(list(new_tags_idx_to_emb.values()), list(new_tags_idx_to_emb.keys()))

        # 생성된 모델 pickle 형태로 변환
        pickle.dump(p_title, open('model/p_title.pickle', 'wb') )
        pickle.dump(p_tags, open('model/p_tags.pickle', 'wb') )
        pickle.dump(p_new_tags, open('model/p_new_tags.pickle', 'wb') )

        print('\nSuccess')

    def w2v_model_make(self, neg = 10):

        w2v_model = Word2Vec(sentences = new_tags,vector_size=100,window=5,min_count=0,workers=6,sg=0,negative=neg)
        w2v_model.save('/home/ubuntu/workspace/jh/project/model/w2v.model')

        # new_tags
        new_tags_emb = []
        for i in train['new_tags'].tolist():
            a = []
            try:
                for j in i :
                    vec = w2v_model.wv.get_vector(j)
                    a.append(vec)
                # x.append(a)
                vec_ave = (sum(a))/len(a)
                new_tags_emb.append(vec_ave)
            except ZeroDivisionError:
                new_tags_emb.append(np.zeros(100))



        new_tags_idx_to_emb = {}
        for idx, emb in enumerate(new_tags_emb):
            if emb.sum() != 0:
                new_tags_idx_to_emb[idx] = emb



        # tags
        tags_emb = []
        for i in train['tags'].tolist():
            a = []
            try:
                for j in i :
                    vec = w2v_model.wv.get_vector(j)
                    a.append(vec)
                # x.append(a)
                vec_ave = (sum(a))/len(a)
                tags_emb.append(vec_ave)
            except ZeroDivisionError and KeyError:
                tags_emb.append(np.zeros(100))



        tags_idx_to_emb = {}
        for idx, emb in enumerate(tags_emb):
            if emb.sum() != 0:
                tags_idx_to_emb[idx] = emb



        # title
        titles_emb = []
        for i in train['plylst_title'].tolist():
            a = []
            try:
                for j in i :
                    vec = w2v_model.wv.get_vector(j)
                    a.append(vec)
                # x.append(a)
                vec_ave = (sum(a))/len(a)
                titles_emb.append(vec_ave)
            except ZeroDivisionError and KeyError:
                titles_emb.append(np.zeros(100))


        titles_idx_to_emb = {}
        for idx, emb in enumerate(tags_emb):
            if emb.sum() != 0:
                titles_idx_to_emb[idx] = emb

        dim = w2v_model.wv.vector_size

        # new_tags
        p_new_tags = hnswlib.Index(space = 'cosine', dim = 100)
        p_new_tags.init_index(max_elements = len(new_tags_idx_to_emb), 
                        ef_construction = 100, M = 16, random_seed = 100)
        p_new_tags.add_items(list(new_tags_idx_to_emb.values()), list(new_tags_idx_to_emb.keys()))


        # tags
        p_tags = hnswlib.Index(space = 'cosine', dim = 100)
        p_tags.init_index(max_elements = len(tags_idx_to_emb), 
                        ef_construction = 100, M = 16, random_seed = 100)
        p_tags.add_items(list(tags_idx_to_emb.values()), list(tags_idx_to_emb.keys()))


        # title
        p_titles = hnswlib.Index(space = 'cosine', dim = 100)
        p_titles.init_index(max_elements = len(titles_idx_to_emb), 
                        ef_construction = 100, M = 16, random_seed = 100)
        p_titles.add_items(list(titles_idx_to_emb.values()), list(titles_idx_to_emb.keys()))

        pickle.dump(p_tags, open('/home/ubuntu/workspace/jh/project/model/w2v_p_tags.pickle', 'wb') )
        pickle.dump(p_titles, open('/home/ubuntu/workspace/jh/project/model/w2v_p_titles.pickle', 'wb') )
        pickle.dump(p_new_tags, open('/home/ubuntu/workspace/jh/project/model/w2v_p_new_tags.pickle', 'wb') )
        print('\nSuccess')