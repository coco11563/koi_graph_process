
import pymysql.cursors
import keyword_objects as kw
import importlib
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

def processApplication(year, db, old_process = False, process_null = False):
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
        if not process_null and (title_zh is None  or keyword_zh is None or abstract_zh is None or research_field is None or applyid is None):
            continue
        else :
            list_obj.append(kw.application(applyid,
                                           title_zh,
                                           abstract_zh,
                                           keyword_zh,
                                           research_field))
    return list_obj

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

    conn = pymysql.connect(host='10.0.202.18',
                           user='root',
                           password='',
                           # _xiaomenG789O
                           db='application_processed',
                           charset='utf8mb4',
                           cursorclass=pymysql.cursors.DictCursor)
    make_all_raw_word_freq_map(2018, conn)