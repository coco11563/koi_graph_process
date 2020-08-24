import dgl as graph
import torch as torch
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


class GraphBase(object) :
    def __init__(self, appid_appid_e, appid_ros_e, appid_app_edge ,ros_application_e,
                 application_keyword_e, id_index_map ,index_id_map, index_ros_map, index_application_map,
                 index_keyword_map, application_dict, retro = False):
        # for seperate usage
        self.id_index_map = id_index_map
        self.aae = appid_appid_e
        self.are = appid_ros_e
        self.aiae = appid_app_edge
        self.rae = ros_application_e
        self.ake = application_keyword_e
        self.iim = index_id_map
        self.irm = index_ros_map
        self.iam = index_application_map
        self.ikm = index_keyword_map
        self.application_dict = application_dict
        self.undirected = retro
        self.appid_set = self.append_edge_to_set(self.aae)
        # if there is only one apply id
        self.appid_set = self.append_edge_to_set(self.are, s_flag=False ,set_=self.appid_set)
        self.appid_set = self.append_edge_to_set(self.aiae, s_flag=False, set_=self.appid_set)

        self.ros_set = self.append_edge_to_set(self.are, f_flag=False)

        self.application_set = self.append_edge_to_set(self.aiae, f_flag=False)
        self.application_set = self.append_edge_to_set(self.ake, s_flag=False, set_=self.application_set)

        self.keyword_set = self.append_edge_to_set(self.ake, f_flag=False)
        self.ctx = None
        if torch.cuda.is_available() :
            self.ctx = torch.device('cuda:0')
        else :
            self.ctx = torch.device('cpu')
        if not self.undirected :
            self.rel_dict = {
                'upon' : 0,
                'contain' : 1,
                'a_include' : 2,
                'b_include' : 3,
                'use' : 4
            }
        else :
            self.rel_dict = {
                'upon': 0,
                'below' : 1,
                'been_contain' : 2,
                'contain': 3,
                'been_a_include' : 4,
                'a_include': 5,
                'been_b_include' : 6,
                'b_include': 7,
                'use': 8,
                'been_use' : 9
            }

    def retro_list(self, li : list):
        ret = []
        for i in li :
            ret.append((i[1], i [0]))
        return ret

    def rel_count(self) :
        if self.undirected :
            return 10
        else :
            return 5

    def get_whole_graph (self, use_feature = False):
        num_dict = dict()
        num_dict['apply_id'] = len(self.appid_set)
        num_dict['ros'] = len(self.ros_set)
        num_dict['application'] = len(self.application_set)
        num_dict['keyword'] = len(self.keyword_set)
        # 在这里可以看出来，节点的类别是不重要的，因为不同的节点类别之间的关系都是不同的，同时若为undirected的话，回去部分的关系也是不同的
        if self.undirected :
            g = graph.heterograph({
            ('n', 'upon', 'n'): self.aae,
            ('n', 'contain', 'n'): self.are,
            ('n', 'a_include', 'n'): self.aiae,
            ('n', 'b_include', 'n'): self.rae,
            ('n', 'use', 'n'): self.ake,
            ('n', 'below', 'n'): self.retro_list(self.aae),
            ('n', 'been_contain', 'n'): self.retro_list(self.are),
            ('n', 'been_a_include', 'n'): self.retro_list(self.aiae),
            ('n', 'been_b_include', 'n'): self.retro_list(self.rae),
            ('n', 'been_use', 'n'): self.retro_list(self.ake),
        })
        else :
            g = graph.heterograph({
                ('n', 'upon', 'n'): self.aae,
                ('n', 'contain', 'n'): self.are,
                ('n', 'a_include', 'n'): self.aiae,
                ('n', 'b_include', 'n'): self.rae,
                ('n', 'use', 'n'): self.ake,
            })
        self.graph_size = len(self.appid_set) + len(self.ros_set) + len(self.application_set) + len(self.keyword_set)
        return g

    def get_unigraph (self):
        g = self.get_whole_graph()
        return graph.to_homo(g)

    def get_split_appid_map (self, appid, recursive = True) :
        aae_edge = set()
        applyid_set = set()
        are_edge = []
        ros_set = set()
        aiae_edge = []
        rae_edge = []
        ake_edge = []
        keyword_set = set()
        application_id_set = set()
        # application是用hash形式存储作为index
        applyid_entity_set = set()
        id_index_map = dict()
        iim = dict()
        iam = dict()
        irm = dict()
        ikm = dict()
        start_index = self.id_index_map[appid]
        applyid_entity_set.add(appid)
        applyid_set.add(start_index)
        new_start_id = 0
        old_new_map = dict()
        old_new_map[start_index] = new_start_id
        iim[new_start_id] = self.iim[start_index]
        id_index_map[self.iim[start_index]] = new_start_id
        # 有一个孤儿节点，怀疑是第一个节点
        new_start_id += 1
        if recursive :
            # 获取这个appid与其所有的子appid相关的子树
            flag = True
            while(flag) :
                flag = False
                tmp_set = set()
                for f,son in self.aae :
                    if applyid_set.__contains__(f) and not applyid_set.__contains__(son) : # son 为上次循环未找到的节点
                        flag = True # 找到新applyid， 下次会寻找这个appid下属节点
                        if not tmp_set.__contains__(son) : # 节点尚未注册
                            tmp_set.add(son)
                            old_new_map[son] = new_start_id
                            iim[new_start_id] = self.iim[son]
                            id_index_map[self.iim[son]] = new_start_id
                            # print(self.iim[son] , '=>' , new_start_id)
                            new_start_id += 1
                        aae_edge.add((old_new_map[f], old_new_map[son]))
                        applyid_entity_set.add(self.iim[son])
                applyid_set = applyid_set.union(tmp_set)
        # print(applyid_set)
        for f, son in self.are:
            if applyid_set.__contains__(f):
                if not ros_set.__contains__(son) :
                    ros_set.add(son)
                    old_new_map[son] = new_start_id
                    irm[new_start_id] = self.irm[son]
                    new_start_id += 1
                are_edge.append((old_new_map[f], old_new_map[son]))
        for f, son in self.aiae:
            if applyid_set.__contains__(f):
                if not application_id_set.__contains__(son) :
                    application_id_set.add(son)
                    old_new_map[son] = new_start_id
                    iam[new_start_id] = self.iam[son]
                    new_start_id += 1
                aiae_edge.append((old_new_map[f], old_new_map[son]))
        for f, son in self.rae:
            if ros_set.__contains__(f) and \
                    applyid_entity_set.__contains__(
                        self.application_dict[
                            self.iam[son]] # ???
                                .applyid):
                if not application_id_set.__contains__(son) :
                    application_id_set.add(son)
                    old_new_map[son] = new_start_id
                    iam[new_start_id] = self.iam[son]
                    new_start_id += 1
                rae_edge.append((old_new_map[f], old_new_map[son]))
        for f, son in self.ake:
            if application_id_set.__contains__(f) :
                if not keyword_set.__contains__(son):
                    keyword_set.add(son)
                    old_new_map[son] = new_start_id
                    ikm[new_start_id] = self.ikm[son]
                    new_start_id += 1
                ake_edge.append((old_new_map[f], old_new_map[son]))
        return GraphBase(list(aae_edge), are_edge, aiae_edge, rae_edge, ake_edge, id_index_map ,iim, irm, iam, ikm,
                         application_dict=self.application_dict, retro = self.undirected)

    def get_split_appid_graph(self, appid, recursive=True):
        gb = \
            self.get_split_appid_map(appid, recursive)
        return gb.get_whole_graph()

    # f_flag : engage father into set
    # s_flag : engage son into set
    def append_edge_to_set(self, edge, set_ = None, f_flag = True, s_flag = True):
        inner_set = set()
        if set_:  # set is not none
            inner_set = set_
        for s,f in edge :
            if f_flag :
                inner_set.add(s)
            if s_flag :
                inner_set.add(f)
        return inner_set

    def info(self):
        return 'this scholar graph contains \n' \
               + 'apply_code : ' + str(len(self.appid_set)) + '\n' \
               + 'ros : ' + str(len(self.ros_set)) + '\n' \
               + 'application : ' + str(len(self.application_set)) +  '\n' \
               + 'keyword : ' + str(len(self.keyword_set)) + '\n'
    def node_count(self):
        return len(self.appid_set) + len(self.ros_set) + len(self.application_set) + len(self.keyword_set)

    def draw(self):
        if self.node_count() > 500 :
            print("too many node to draw")
            pass
        else :
            nx.draw(self.get_unigraph().to_networkx(), with_labels=True
                , node_shape='.')
            plt.show()

    def sample_test(self, sample_per : float):
        data = np.random.random(len(self.application_set))
        application_li = list(self.application_set.copy())
        mask = data > sample_per
        train_application = set()
        test_application = set()
        test_e = []
        aiae = []
        rae = []
        index = 0
        for flag in mask :
            if flag :
                train_application.add(application_li[index])
            else :
                test_application.add(application_li[index])
            index += 1

        for (f,son) in self.aiae :
            if train_application.__contains__(son) :
                aiae.append((f,son))
            else :
                test_e.append((f,0 ,son)) # 0 represent applyid to application rel type
        for (f,son) in self.rae :
            if train_application.__contains__(son) :
                rae.append((f,son))
            else :
                test_e.append((f, 1 , son)) # 1 represent applyid to application rel type
        return GraphBase(self.aae, self.are, aiae, rae, self.ake, self.id_index_map ,self.iim, self.irm, self.iam, self.ikm,
                         application_dict=self.application_dict, retro=self.undirected), \
               test_application, test_e

    def negative_test_sampling_are (self, sample_rate = 0.5, negative_rate = 10) :
        edge_array = np.array(self.rae)
        src = edge_array[:,0]
        dst = edge_array[:,1]
        rel = np.ones(src.shape) * self.rel_dict['b_include'] # this rel is no.3
        relabeled_edges = np.stack((src, rel, dst)).transpose()
        relabeled_pair = np.stack((src, dst)).transpose()
        # print(edge_array.shape)
        num_entity = len(self.application_set)
        # are edge size is same as application number
        sample_size = int(sample_rate * num_entity)
        # 随机sample若干application节点 （50%）
        chosen_are_tag =  np.random.choice(np.arange(num_entity), sample_size, replace=False)
        chosen_are = relabeled_pair[chosen_are_tag]
        neg_sample, label = self.negative_sampling_are(relabeled_edges, negative_rate)
        return GraphBase(self.aae, self.are, self.aiae, list(chosen_are), self.ake, self.id_index_map, self.iim, self.irm, self.iam,
                         self.ikm, application_dict=self.application_dict, retro=self.undirected), \
               neg_sample, label

    # pos_sample : positive samples
    def negative_sampling_are(self, pos_samples, negative_rate):
        size_of_batch = len(pos_samples)
        # 列方向重复1次 行方向重复10次
        labels = np.zeros(size_of_batch * (negative_rate + 1), dtype=np.float32)
        # 将正例的label设置为 1
        labels[: size_of_batch] = 1
        for i in pos_samples:
            pos_sub = int(i[0])
            neg_samples = np.tile(i, (negative_rate, 1))
            neg_ros_set = list(self.ros_set.difference({pos_sub}))
            values = np.array([neg_ros_set[i] for i in np.random.randint(len(neg_ros_set), size=negative_rate)])
            choices = np.random.uniform(size=negative_rate)
            subj = choices > 0  # all will be replace
            neg_samples[subj, 0] = values[subj]
            pos_samples = np.concatenate((pos_samples, neg_samples))
        return pos_samples, labels

    def comp_deg_norm(self):
        g = self.get_unigraph().local_var()
        in_deg = g.in_degrees(range(g.number_of_nodes())).float().numpy()
        norm = 1.0 / in_deg
        norm[np.isinf(norm)] = 0
        return torch.tensor(norm)

    def node_norm_to_edge_norm(self, node_norm):
        g = self.get_whole_graph()
        # convert to edge norm
        g.nodes['n'].data['norm'] = node_norm
        for e_type in g.etypes :
            g.apply_edges(lambda edges : {'norm' : edges.dst['norm']} , etype=e_type)
        return g