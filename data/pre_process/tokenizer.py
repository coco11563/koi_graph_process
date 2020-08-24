import jieba
import os
import re
import keyword_objects as kw
def remove_jieba_cache():
    if os.path.exists("/tmp/jieba.cache") :
        print('检测到缓存文件，清除中')
        os.remove("/tmp/jieba.cache")
        print('缓存移除完成')


def init_jieba_dict(word_tuple, reload = False):
    if reload :
        remove_jieba_cache()
    jb = jieba.Tokenizer(dictionary=jieba.DEFAULT_DICT)
    for tu in word_tuple:
        jb.add_word(tu[0], freq=tu[1])
    return jb


def tokenize(corpora_list: list, cut_all: bool, word_tuple: list, reload = False):
    jb = init_jieba_dict(word_tuple, reload)
    token_list = []
    for cor in corpora_list:
        cor_list = []
        for tok_cor in cor :
            this_tok = []
            for token in (jb.cut(tok_cor, cut_all)):
                this_tok.append(token)
            cor_list.append(this_tok)
        token_list.append(cor_list)
    return token_list


# get corpora source by id
def get_corpora_source(id, tree_set : kw.apply_id_tree, list_application : list) :
    id_set = set(tree_set.get(id))
    corpora = []
    for application in list_application :
        application_id = application.applyid
        if application_id in id_set:
            str = ""
            for keys in application.keyword :
                # str += "<" + keys + ":" + application.research_field + ":" + application.applyid +  ">"
                str += keys
            corpora.append(str)
            corpora.append(application.title)
            abstract = application.abstract.replace("，", " ")
            abstract = abstract.split("。")
            for sentence in abstract :
                corpora.append(sentence)
            corpora.append(application.research_field)
    return corpora

def get_all_corpora_source(list_application : list) :
    corpora = []
    for application in list_application :
        str = ""
        for keys in application.keyword :
            # str += "<" + keys + ":" + application.research_field + ":" + application.applyid +  ">"
            str += keys
        corpora.append(str)
        corpora.append(application.title)
        abstract = application.abstract.replace("，", " ")
        abstract = abstract.split("。")
        for sentence in abstract :
            corpora.append(sentence)
        corpora.append(application.research_field)
    return corpora

def find_chinese(file):
    pattern = re.compile(r'[^\u4e00-\u9fa5]')
    return re.sub(pattern, '', file)

def get_corpora_source_for_bert(list_application : list) :
    corpora = []
    for application in list_application :
        str_list = []
        # skip the key words phase
        # for keys in application.keyword :
        #     # str += "<" + keys + ":" + application.research_field + ":" + application.applyid +  ">"
        #     str += keys
        # corpora.append(str)
        str_list.append(application.title.strip())
        abstract = application.abstract.strip().replace("，", " ")
        abstract = abstract.split("。")
        for sentence in abstract :
            if sentence.strip() == '' :
                continue
            str_list.append(find_chinese(sentence))
        str_list.append(application.research_field.strip())
        corpora.append(str_list)
    return corpora