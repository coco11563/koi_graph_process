
import pymysql.cursors
import keyword_objects as kw
import importlib
import copy
importlib.reload(kw)
def initSQLConn():
    connection = pymysql.connect(host='10.0.82.237',
                                 user='root',
                                 password='xiaomenG789O_',
                                 # _xiaomenG789O
                                 db='application_process',
                                 charset='utf8mb4',
                                 cursorclass=pymysql.cursors.DictCursor)
    print(connection.get_server_info())
    print(connection.get_proto_info())
    print(connection.get_host_info())
    return connection

def processKeywordSql(year,connection, old_process = False, process_null = False):
    cursor = connection.cursor()
    if not old_process :
        cursor.execute("SELECT apply_id,zh_title,zh_keyword,zh_abstract,research_field FROM " + str(year) + "_application_new;")
    else :
        cursor.execute("SELECT applyid,zh_title,ckeyword,abstract FROM" + str(year) + "_APPLICATION_OLD")
    data = cursor.fetchall()
    id_tree = kw.apply_id_tree()
    words = set()
    word2code = dict()
    code2word_set = dict()
    for row in data :
        if not old_process :
            title_zh = row['zh_title']
            keyword_zh = row['zh_keyword']
            abstract_zh = row['zh_abstract']
            research_field = row['research_field']
            applyid = row['apply_id']
        else :
            research_field = None
            title_zh = row['zh_title']
            keyword_zh = row['ckeyword']
            abstract_zh = row['abstract']
            applyid = row['applyid']
        if not process_null and (title_zh is None  or keyword_zh is None or abstract_zh is None or research_field is None or applyid is None):
            continue
        else :
            keywords_zh = keyword_zh.split('；')
            for word in keywords_zh :
                keyword = kw.keyword(word, research_field, applyid)
                words.add(keyword)
                id_tree.add(applyid)
                if code2word_set.__contains__(applyid):
                    code2word_set.get(applyid).add(keyword)
                else:
                    code2word_set.update({applyid:set()})
                    code2word_set.get(applyid).add(keyword)
    return words, word2code, id_tree, code2word_set

def processRS2Code(year,connection):
    cursor = connection.cursor()
    cursor.execute("SELECT apply_id,zh_title,zh_keyword,zh_abstract,research_field FROM " + str(year) + "_application_new;")
    data = cursor.fetchall()
    rs2code = dict()
    for row in data :
        research_field = row['research_field']
        applyid = row['apply_id']
        rs2code.update({research_field:applyid})
    return rs2code

def processApplyId(applylist : list) :
    return [i.applyid for i in applylist]

def isEndCodeSimple(s) :
    return len(s) == len('A010101')

def isSecCodeSimple(s) :
    return len(s) == len('A0101')

def isFirstCodeSimple(s) :
    return len(s) == len('A01')

def isHeadCodeSimple(s) :
    return len(s) == len('A')

def isRootSimple(s) :
    return s == 'root'

def EndCode2CodeSimple(code) :
    return code[:5]

def getLastCodeSimple(code) :
    if isHeadCodeSimple(code) : return code
    else :
        l = len(code)
        end_index = l - 2
        return code[:end_index]

def getCodeLevel(code) :
    if isEndCodeSimple(code) : return 5
    elif isSecCodeSimple(code) : return 4
    elif isFirstCodeSimple(code) : return 3
    elif isHeadCodeSimple(code) : return 2
    elif isRootSimple(code) : return 1
    else : return 0

# basic test pass
# 输入appid list 返回appid 对应的index
def processApplyIdList(applyid_list : list, start_id) :
    unique_set = set()
    used_code_set = set()
    for id in applyid_list :
        unique_set.add(id)
    start_id_ = start_id
    id_index_map = dict()
    for id in unique_set :
        id_index_map[id] = start_id_
        start_id_ += 1
    item = copy.deepcopy(id_index_map).items()
    for k,v in item :
        if isEndCodeSimple(k) : #确认是三级代码
            pre_code = EndCode2CodeSimple(k)
            used_code_set.add(pre_code)
            if id_index_map.__contains__(pre_code) :
                id_index_map[k] = id_index_map[pre_code] # 该末级代码指向上级

            else :
                id_index_map[pre_code] = start_id_ # no add
                start_id_ += 1
                id_index_map[k] = id_index_map[pre_code]
        else:
            used_code_set.add(k)
    item = id_index_map.items()
    # 精简code使其从0开始
    id_index_map_ = dict()
    start = 0
    for k,v in item :
        if id_index_map_.__contains__(id_index_map[k]) :
            id_index_map[k] = id_index_map_[v]
        else :
            id_index_map_[v] = start
            start += 1
            id_index_map[k] = id_index_map_[v]
    index_id_map = dict()
    for code in used_code_set :
        index_id_map[id_index_map[code]] = code
    return id_index_map, index_id_map, len(used_code_set) - 1

def processApplyIdListWithLimit(applyid_list : list, start_id, code_limit = 3) :
    unique_set = set()
    used_code_set = set()
    for id in applyid_list :
        unique_set.add(id)
    start_id_ = start_id
    id_index_map = dict()
    for id in unique_set :
        id_index_map[id] = start_id_
        start_id_ += 1
    item = copy.deepcopy(id_index_map).items()
    for k,v in item :
        code = k
        if getCodeLevel(code) > code_limit :
            while getCodeLevel(code) > code_limit :
                code = getLastCodeSimple(code) # 取得符合标准的code
            used_code_set.add(code)
            if id_index_map.__contains__(code):
                id_index_map[k] = id_index_map[code]  # 该末级代码指向上
            else:
                id_index_map[code] = start_id_  # no add
                start_id_ += 1
                id_index_map[k] = id_index_map[code]
        else:
            used_code_set.add(k)
    item = id_index_map.items()
    # 精简code使其从0开始
    id_index_map_ = dict()
    start = 0
    for k,v in item :
        if id_index_map_.__contains__(id_index_map[k]) :
            id_index_map[k] = id_index_map_[v]
        else :
            id_index_map_[v] = start
            start += 1
            id_index_map[k] = id_index_map_[v]
    index_id_map = dict()
    for code in used_code_set :
        index_id_map[id_index_map[code]] = code
    return id_index_map, index_id_map, len(used_code_set) - 1
# old_process False process from xx_application_new or xx_application_old
# process_null False process null attribute? False will skip all row contains null attribute
def processApplication(year, db, old_process = False, process_null = False, filter = None, quest = None):
    cursor = db.cursor()
    if not old_process:
        cursor.execute("SELECT apply_id,zh_title,zh_keyword,zh_abstract,research_field FROM " + str(year) + "_application_new;")
    else:
        cursor.execute("SELECT applyid,zh_title,ckeyword,abstract FROM" + str(year) + "_APPLICATION_OLD")
    data = cursor.fetchall()
    print(len(data))
    list_obj=[]
    for row in data :
        if not old_process:
            # only new table have research_field
            title_zh = row['zh_title']
            keyword_zh = row['zh_keyword']
            abstract_zh = row['zh_abstract']
            research_field = row['research_field']
            applyid = row['apply_id']
        else:
            research_field = None
            title_zh = row['zh_title']
            keyword_zh = row['ckeyword']
            abstract_zh = row['abstract']
            applyid = row['applyid']
        if not process_null \
                and (title_zh is None or keyword_zh is None or abstract_zh is None or research_field is None or applyid is None):
            # if this process is focus on new process, there might be few research_field is none,
            # this \*if judge*\ will give them a chance
            if not old_process and title_zh is not None and keyword_zh is not None and abstract_zh is not None and applyid is not None :
                # print(applyid)
                list_obj.append(kw.application(applyid,
                                               title_zh,
                                               abstract_zh,
                                               keyword_zh,
                                               research_field))
        else :
            list_obj.append(kw.application(applyid,
                                           title_zh,
                                           abstract_zh,
                                           keyword_zh,
                                           research_field))
    if filter :
        ret = []
        for i in list_obj :
            if filter(i, quest) :
                ret.append(i)
        list_obj = ret
    return list_obj
def applyid_filter(app : kw.application, quest) :
    applyid = app.applyid
    if len(applyid) > len(quest) :
        q_length = len(quest)
        if applyid[0:q_length] == quest :
            return True
    elif len(quest) > len(applyid) :
        q_length = len(applyid)
        if quest[0:q_length] == applyid :
            return True
    else :
        if quest == applyid :
            return True
    return False
def make_word_freq_map (year, db) :
    cursor = db.cursor()
    cursor.execute(
        "SELECT * FROM " + str(year) + "_keyword_count;")
    data = cursor.fetchall()
    print(len(data))
    dict_obj = dict()
    for row in data :
        k_f = int(row['title_f']) + int(row['keyword_f']) + int(row['abstract_f'])
        k_all_f = int(row['title_all']) + int(row['keyword_all']) + int(row['abstract_all'])
        keyword = row['keyword']
        research_field = row['research_field']
        applyid = row['applyid']
        if applyid is None :
            continue
        else :
            keyword = kw.keyword(keyword, research_field, applyid, frequent = k_f, frequent_all = k_all_f)
            dict_obj.update({keyword : (k_f, k_all_f)})
    return dict_obj

def make_all_raw_word_freq_map (year, db) :
    cursor = db.cursor()
    cursor.execute(
        "SELECT * FROM " + str(year) + "_keyword_count;")
    data = cursor.fetchall()
    print(len(data))
    dict_obj = dict()
    dict_set = dict()
    for row in data :
        k_all_f = int(row['title_all']) + int(row['keyword_all']) + int(row['abstract_all'])
        keyword = row['keyword'].replace("？","").replace(" ", "")
        applyid = str(row['applyid'])[0]
        if not dict_set.__contains__(applyid) :
            dict_set[applyid] = {keyword}
            if not dict_obj.__contains__(keyword) :
                dict_obj[keyword] = k_all_f
            else :
                tmp = dict_obj[keyword]
                dict_obj[keyword] = tmp + k_all_f
        else :
            set = dict_set[applyid]
            if not set.__contains__(keyword) :
                set.add(keyword)
                if not dict_obj.__contains__(keyword) :
                    dict_obj[keyword] = k_all_f
                else :
                    tmp = dict_obj[keyword]
                    dict_obj[keyword] = tmp + k_all_f
            else :
                continue
    return dict_obj

if __name__ == '__main__':
    import pymysql.cursors
    conn = initSQLConn()
    count = 0
    # make_all_raw_word_freq_map(2018, conn)
    appli = processApplication(2019, conn, filter=applyid_filter, quest='F01')
    print(len(appli))