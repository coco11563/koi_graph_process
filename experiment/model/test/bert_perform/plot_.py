from gensim.models import word2vec
import mysql_utils as mysql
import numpy as np
from openTSNE import TSNE

class ModelBase:

    year_model = dict()
    year_tree = dict()
    year_model_2d = dict()

    def __init__(self, file_names, conn):
        for file_name in file_names:
            self.year_model[file_name[0]] = word2vec.Word2Vec.load(file_name[1]).wv
            # trans to 2d
            # self.year_model_2d[file_name[0]] = self.mk_nparray([item[1] for item in word2vec.Word2Vec.load(file_name[1]).wv])
            _, _, _, id_tree, code2word = mysql.processKeywordSql(file_name[0], conn)
            self.year_tree[file_name[0]] = (id_tree,code2word)

    def get_model(self, year):
        if self.year_model.__contains__(year):
            return self.year_model[year]
        else:
            return None

    def get_tree(self, year):
        if self.year_model.__contains__(year):
            return self.year_model[year]
        else:
            return None

    def get_all_vector(self, code, year, not_contain = None):
        tree = self.year_tree.get(year)[0]
        code2word = self.year_tree.get(year)[1]
        wv = self.year_model.get(year)
        code_list = tree.get(code)
        print(len(code_list))
        if not_contain is not None :
            if type(not_contain) is list :
                for code_ in not_contain :
                    not_contain_list = tree.get(code_)
                    for not_code_ in not_contain_list :
                        code_list.remove(not_code_)
            if type(not_contain) is str:
                not_contain_list = tree.get(not_contain)
                for not_code_ in not_contain_list:
                    code_list.remove(not_code_)
        word_vectors = []
        word_list = []
        dis_set = set()
        print(type(code2word))
        for code in code_list :
            if code2word.__contains__(code):
                for word in code2word[code] :
                    if wv.__contains__(word.name) and not dis_set.__contains__(word.name):
                        word_list.append(word.name)
                        word_vectors.append(wv[word.name])
                        dis_set.add(word.name)
        return word_list, word_vectors

    def get_all_vector_2d(self, code, year, not_contain = None):
        tree = self.year_tree.get(year)[0]
        code2word = self.year_tree.get(year)[1]
        wv = self.tsne(year)
        code_list = tree.get(code)
        print(len(code_list))
        if not_contain is not None :
            if type(not_contain) is list :
                for code_ in not_contain :
                    not_contain_list = tree.get(code_)
                    for not_code_ in not_contain_list :
                        code_list.remove(not_code_)
            if type(not_contain) is str:
                not_contain_list = tree.get(not_contain)
                for not_code_ in not_contain_list:
                    code_list.remove(not_code_)
        word_vectors = []
        word_list = []
        dis_set = set()
        print(type(code2word))
        for code in code_list :
            if code2word.__contains__(code):
                for word in code2word[code] :
                    if wv.__contains__(word.name) and not dis_set.__contains__(word.name):
                        word_list.append(word.name)
                        word_vectors.append(wv[word.name])
                        dis_set.add(word.name)
        return word_list, word_vectors

    # year_model_2d = dict()
    # default use eu distance
    # default use eu distance
    def embed(self,X):
        X_embedded = TSNE(n_jobs=-2).fit(X)
        return X_embedded

    def mk_nparray(self,vec):
        if len(vec) == 0:
            return np.array([])
        vector = np.array(vec)
        re = self.embed(vector)
        return re

    def tsne(self, year):
        if self.year_model_2d.__contains__(year) :
            return self.year_model_2d[year]
        tree = self.year_tree.get(year)[0]
        code2word = self.year_tree.get(year)[1]
        wv = self.year_model.get(year)
        word_vectors = []
        word_list = []
        dis_set = set()
        print(type(code2word))
        for code in tree.get_all() :
            if code2word.__contains__(code):
                for word in code2word[code] :
                    if wv.__contains__(word.name) and not dis_set.__contains__(word.name):
                        word_list.append(word.name)
                        word_vectors.append(wv[word.name])
                        dis_set.add(word.name)
        print("start to tsne the data: " + year)
        word_vectors = self.mk_nparray(word_vectors)
        print("done: "+ year)
        _2d_dict = dict()
        ind = 0
        for w in word_list :
            _2d_dict[w] = word_vectors[ind]
            ind += 1
        self.year_model_2d[year] = _2d_dict
        return _2d_dict

