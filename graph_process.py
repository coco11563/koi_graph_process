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
        id_index_map[app_hash] = app_id
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
                if apply_ros_set.__contains__(ros):
                    ros_id = ros_index_map[ros]
                    appid_ros_edge.append((appid_id, ros_id))
                    app_ros_edge.append((ros_id, app_id))
                else:
                    ros_id = start_id
                    start_id += 1
                    ros_index_map[ros] = ros_id
                    index_ros_map[ros_id] = ros
                    apply_ros_set.add(ros)
                    appid_ros_edge.append((appid_id, ros_id))
                    app_ros_edge.append((ros_id, app_id))
        else:
            appid_app_edge.append((appid_id, app_id))
    return index_keyword_map, index_ros_map, index_app_map, appid_ros_edge, appid_app_edge, app_kw_edge, app_ros_edge, app_dict, start_id


#
def process_year_graph(year, process_null=True, process_old=False):
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
    application_list = mysql.processApplication(year, conn)
    index_keyword_map, index_ros_map, index_app_map, appid_ros_edge, appid_app_edge, app_kw_edge, app_ros_edge, app_dict, start_id_ = \
        apply_application_graph(application_list, start_id, id_index_map, not process_old)
    print('done init {} nodes from application'.format(start_id_ - 1))
    return appid_edge, appid_ros_edge, appid_app_edge, app_ros_edge, app_kw_edge, \
           id_index_map, index_id_map, index_keyword_map, index_ros_map, index_app_map, \
           app_dict


# this can only print meta path
import pygraphviz as pgv


def plot_graph(nxg):
    ag = pgv.AGraph(strict=False, directed=True)
    for u, v, k in nxg.edges(keys=True):
        ag.add_edge(u, v, label=k)
    ag.layout('dot')
    ag.draw('graph.png')

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

