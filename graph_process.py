import mysql_utils as mysql
import numpy as np
import obj.GraphBase as GraphBase
import utils
import torch as torch
# 从RMDBS中取出图相关的数据，并将其转化成边关系
import importlib
importlib.reload(mysql)
# 父节点指向子节点
# id => app id
# index => graph id
# if simple = true, all the EndCode will be ignored
def apply_id_graph(root, start_id, edge, index_id_map, id_index_map, root_apply_id):
    for keys in root:
        id_index_map[keys] = start_id
        index_id_map[start_id] = keys
        start_id += 1
        # if start_id == 1101 :
        #     print(root)
        if len(root[keys].keys()) > 0:  # 如果keys存在子节点
            # start_id += 1
            # maybe IS THE ISSUE
            edge.append((id_index_map[root_apply_id], id_index_map[keys]))
            root_ = root[keys]
            start_id = apply_id_graph(root_, start_id, edge, index_id_map, id_index_map, keys)
        else:  # 如果keys不存在子节点
            # print(id_index_map[root_apply_id], '|=>' , id_index_map[keys])
            edge.append((id_index_map[root_apply_id], id_index_map[keys]))
    return start_id


def apply_application_graph(application_list, start_id, id_index_map, useros=True):
    app_dict = dict()
    keyword_index_map = dict()
    ros_index_map = dict()
    # 将图内的id 转换为keyword ros 和app hash
    index_keyword_map = dict()
    index_ros_map = dict()
    index_app_map = dict()
    appid_ros_map_set = dict()
    app_index_map = dict()

    keyword_set = set()
    apply_ros_set = set()

    app_kw_edge = []
    appid_ros_edge = []
    appid_app_edge = []
    app_ros_edge = []
    for application in application_list:
        if application is None: continue
        app_hash = application.__hash__()
        app_dict[app_hash] = application
        keywords = application.keyword
        app_id = start_id
        index_app_map[app_id] = app_hash
        app_index_map[app_hash] = app_id
        start_id += 1
        for kw in keywords:
            # if kw == '' : continue # 去除空关键字
            if keyword_set.__contains__(kw) :
                kw_id = keyword_index_map[kw]
                app_kw_edge.append((app_id, kw_id))
            else:
                keyword_index_map[kw] = start_id
                index_keyword_map[start_id] = kw
                app_kw_edge.append((app_id, start_id))
                keyword_set.add(kw)
                start_id += 1
        ros = application.research_field
        appid_id = id_index_map[application.applyid]
        if useros: # is this cursor is to table_old then no ros will be provided
            if ros == None :
                    # or ros == '': # 去除空ros
                appid_app_edge.append((appid_id, app_id))
            else:
                # ros是string形式的object
                if apply_ros_set.__contains__(ros): # 判断是否需要为该ros赋新id
                    ros_id = ros_index_map[ros]
                    if appid_ros_map_set.__contains__(appid_id) :
                        ros_set = appid_ros_map_set[appid_id]
                        if not ros_set.__contains__(ros_id) :
                            appid_ros_edge.append((appid_id, ros_id))
                            appid_ros_map_set[appid_id].add(ros_id)
                        # else : 边已存在
                    else :
                        appid_ros_map_set[appid_id] = set()
                        appid_ros_map_set[appid_id].add(ros_id)
                        appid_ros_edge.append((appid_id, ros_id))
                    app_ros_edge.append((ros_id, app_id))
                else: # 新出现的ros
                    ros_id = start_id
                    start_id += 1
                    ros_index_map[ros] = ros_id
                    index_ros_map[ros_id] = ros
                    apply_ros_set.add(ros)
                    if appid_ros_map_set.__contains__(appid_id) :
                        ros_set = appid_ros_map_set[appid_id]
                        if not ros_set.__contains__(ros_id) :
                            appid_ros_edge.append((appid_id, ros_id))
                            appid_ros_map_set[appid_id].add(ros_id)
                        # else : 边已存在
                    else :
                        appid_ros_map_set[appid_id] = set()
                        appid_ros_map_set[appid_id].add(ros_id)
                        appid_ros_edge.append((appid_id, ros_id))
                    # appid_ros_edge.append((appid_id, ros_id))
                    app_ros_edge.append((ros_id, app_id))
        else:
            appid_app_edge.append((appid_id, app_id))
    return index_keyword_map, index_ros_map, index_app_map, appid_ros_edge, appid_app_edge, app_kw_edge, app_ros_edge, app_dict, \
           start_id, keyword_index_map, ros_index_map,app_index_map



#
def process_year_graph(year, process_null=True, process_old=False, filter = None, quest= None):
    conn = mysql.initSQLConn()
    _, _, id_tree, code2word = mysql.processKeywordSql(year, conn, process_old, process_null)
    root = id_tree.root
    start_id = 0
    appid_edge = []
    index_id_map = dict()
    id_index_map = dict()
    root_apply_id = 'root'
    index_id_map[start_id] = root_apply_id
    id_index_map[root_apply_id] = start_id
    start_id += 1
    start_id = apply_id_graph(root, start_id, appid_edge, index_id_map, id_index_map, root_apply_id)
    print('done init {} nodes from apply id'.format(start_id - 1))
    application_list = mysql.processApplication(year, conn, filter=filter, quest=quest)
    index_keyword_map, index_ros_map, index_app_map, appid_ros_edge, appid_app_edge, app_kw_edge, app_ros_edge, app_dict, start_id_ ,_,_,_= \
        apply_application_graph(application_list, start_id, id_index_map, not process_old)
    print('done init {} nodes from application'.format(start_id_ - 1))
    return appid_edge, appid_ros_edge, appid_app_edge, app_ros_edge, app_kw_edge, \
           id_index_map, index_id_map, index_keyword_map, index_ros_map, index_app_map, \
           app_dict
# quit all end code
def process_year_graph_simple(year, process_null=True, process_old=False, code_limit=3, filter = None, quest= None):
    conn = mysql.initSQLConn()
    # make_all_raw_word_freq_map(2018, conn)
    appli = mysql.processApplication(year, conn, old_process=process_old, process_null=process_null , filter=filter, quest=quest)
    appid_li = [i.applyid for i in appli]
    # i = ['A010101', 'A0101', 'B010101']
    id_index_map, index_id_map, _ = mysql.processApplyIdListWithLimit(appid_li, 0, code_limit=code_limit)
    print('apply id now')
    index_id_map, id_index_map, edge, start_id = apply_id_simple(index_id_map, id_index_map)
    print('done init {} nodes from apply id'.format(start_id - 1))
    index_keyword_map, index_ros_map, index_app_map, appid_ros_edge, appid_app_edge, app_kw_edge, app_ros_edge, app_dict, start_id_ , _, _ , _= \
        apply_application_graph(appli, start_id, id_index_map, not process_old)
    print('done init {} nodes from application'.format(start_id_ - 1))
    return edge, appid_ros_edge, appid_app_edge, app_ros_edge, app_kw_edge, \
           id_index_map, index_id_map, index_keyword_map, index_ros_map, index_app_map, \
           app_dict
# 不添加额外连接， 每个只连接到上层
# return indexidmap 讲数字转换为applyid
# id index map 讲未简化的ID转换为index
# edge 返回 appid 边
def apply_id_simple(index_id_map : dict, id_index_map : dict) :
    apply_id_set = id_index_map.keys()
    start_id = len(index_id_map) # 目前位置
    edge = []
    import copy
    index_id_map_ = copy.deepcopy(index_id_map)
    id_index_map_ = copy.deepcopy(id_index_map)
    for s in apply_id_set :
        code = s
        index_pre = id_index_map_[code]
        while(mysql.getCodeLevel(code) >= 0) :
            code = mysql.getLastCodeSimple(code)
            if id_index_map_.__contains__(code) : # 直接找到上层
                edge.append((id_index_map_[code] , index_pre))
                break
            elif mysql.getCodeLevel(code) == 2:
                index_id_map_[start_id] = code
                id_index_map_[code] = start_id
                start_id += 1
                edge.append((id_index_map_[code], index_pre))
                break
                # 若这个code是顶级编码但不属于A-H
            else : continue
    return index_id_map_, id_index_map_, edge, start_id

# this can only print meta path
import pygraphviz as pgv


def plot_graph(nxg):
    ag = pgv.AGraph(strict=False, directed=True)
    for u, v, k in nxg.edges(keys=True):
        ag.add_edge(u, v, label=k)
    ag.layout('dot')
    ag.draw('graph.png')
# turn gb to edge list
def get_graph_edges(graph_base : GraphBase, train = 0.8, test = 0.1, validation = 0.1) :
    if train + test + validation != 1 :
        raise Exception("the train/test/validation percentage is not right")
    app_list = []
    if graph_base.undirected :
        edge_build(graph_base.aae,0,app_list)
        edge_build(graph_base.are,2,app_list)
        edge_build(graph_base.aiae,4,app_list)
        edge_build(graph_base.rae,6,app_list)
        edge_build(graph_base.ake,8,app_list)
        edge_build(graph_base.retro_list(graph_base.aae),1, app_list)
        edge_build(graph_base.retro_list(graph_base.are),3, app_list)
        edge_build(graph_base.retro_list(graph_base.aiae),5, app_list)
        edge_build(graph_base.retro_list(graph_base.rae),7, app_list)
        edge_build(graph_base.retro_list(graph_base.ake),9, app_list)
    else :
        edge_build(graph_base.aae, 0,app_list)
        edge_build(graph_base.are, 1,app_list)
        edge_build(graph_base.aiae, 2,app_list)
        edge_build(graph_base.rae, 3,app_list)
        edge_build(graph_base.ake, 4,app_list)
    edges = np.concatenate(app_list)
    return edges_data(edges, train, test)

def edge_build(edges, rel_no, app_list) :
    if len(edges) != 0 :
        edge_array = np.array(edges, dtype=np.int).transpose()
        print(edge_array.shape)
        print(type(edges[0][0]))
        src = edge_array[0,:]
        dst = edge_array[1,:]
        rel = np.ones(src.shape, dtype=np.int) * rel_no
        relabeled_edges = np.stack((src, rel, dst)).transpose()
        print(type(relabeled_edges[0][0]))
        app_list.append(relabeled_edges)
# random pick edge
class edges_data(object) :
    def __init__(self, edges, train_p, test_p) :
        num_to_generate = edges.shape[0]
        choices = np.random.uniform(size=num_to_generate)
        train_flag = choices <= train_p
        test_flag = (choices > train_p) & (choices <= train_p + test_p)
        validation_flag = choices > (train_p + test_p)
        self.test = edges[test_flag]
        self.train = edges[train_flag]
        self.validation = edges[validation_flag]

def make_test_graph(edges, rel_num) :
    src, rel, dst = edges.transpose()
    uniq_v, edges = np.unique((src, dst), return_inverse=True)
    src, dst = np.reshape(edges, (2, -1))
    relabeled_edges = np.stack((src, rel, dst)).transpose()
    num_node = len(uniq_v)
    test_graph, test_rel, test_norm = utils.build_graph_from_triplets(
            num_node, rel_num, (src, rel, dst))
    test_node_id = uniq_v
    test_rel = torch.from_numpy(test_rel)
    return test_graph, test_node_id, test_rel, test_norm

