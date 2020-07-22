import torch as torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.nn.pytorch as dglModel
# from dgl.nn.pytorch import RelGraphConv
# this model will combine all embedding together(from bottom to the top)
from gensim.models import word2vec
from dgl import function as fn
from dgl.nn.pytorch import utils

from dgl_model import BaseRGCN

semantic_base = word2vec.Word2Vec.load('2019_model')

class ZeroInitEmbeddingLayer(nn.Module):
    # this embedding layer represent semantic information
    def __init__(self, num_nodes, h_dim, i2w_dict):
        super(ZeroInitEmbeddingLayer, self).__init__()
        self.embedding = torch.nn.Embedding(num_nodes, h_dim)
        self.embedding.weight.data = torch.zeros((num_nodes, h_dim))
            # init w2v
        for i in range(num_nodes):
            if i2w_dict.__contains__(i):
                self.embedding.weight[i] = torch.FloatTensor(semantic_base[i2w_dict[i]])

    # squeeze remove useless dim

    def forward(self, g, h):
        return self.embedding(h.squeeze())

class RGCN(BaseRGCN):
    def __init__(self, num_nodes, h_dim, out_dim, num_rels, num_bases, i2w_dict,
                 num_hidden_layers=1, dropout=0,
                 use_self_loop=False, use_cuda=False):
        super(RGCN, self).__init__(num_nodes, h_dim, out_dim, num_rels, num_bases,
                 num_hidden_layers=1, dropout=0,
                 use_self_loop=False, use_cuda=False)
        self.num_nodes = num_nodes
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.num_rels = num_rels
        self.num_bases = None if num_bases < 0 else num_bases
        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout
        self.use_self_loop = use_self_loop
        self.use_cuda = use_cuda
        self.i2w_dict = i2w_dict
        # create rgcn layers
        self.build_model()

    def build_input_layer(self):
        return ZeroInitEmbeddingLayer(self.num_nodes, self.h_dim, self.i2w_dict)

    def build_hidden_layer(self, idx):
        # act = F.relu if idx < self.num_hidden_layers - 1 else None
        return RelGraphConvRefactor(self.h_dim, self.h_dim, self.num_rels, "basis",
                self.num_bases, activation=None, self_loop=False,
                dropout=self.dropout)




class LinkPredict(nn.Module):
    def __init__(self, in_dim, h_dim, num_rels, dict, num_bases=-1,
                 num_hidden_layers=1, dropout=0, use_cuda=False, reg_param=0):
        super(LinkPredict, self).__init__()
        self.rgcn = RGCN(in_dim, h_dim, h_dim, num_rels * 2, num_bases, dict,
                         num_hidden_layers, dropout, use_cuda)
        # self.embedding = self.rgcn.layers[0]
        self.reg_param = reg_param
        self.w_relation = nn.Parameter(torch.Tensor(num_rels, h_dim))
        nn.init.xavier_uniform_(self.w_relation,
                                gain=nn.init.calculate_gain('relu'))

    def calc_score(self, embedding, triplets):
        # DistMult
        s = embedding[triplets[:,0]]
        r = self.w_relation[triplets[:,1]]
        o = embedding[triplets[:,2]]
        score = torch.sum(s * r * o, dim=1)
        return score

    def forward(self, g, h, r, norm):
        return self.rgcn.forward(g, h, r, norm)

    def regularization_loss(self, embedding):
        return torch.mean(embedding.pow(2)) + torch.mean(self.w_relation.pow(2))

    def get_loss(self, g, embed, triplets, labels):
        # triplets is a list of data samples (positive and negative)
        # each row in the triplets is a 3-tuple of (source, relation, destination)
        score = self.calc_score(embed, triplets)
        predict_loss = F.binary_cross_entropy_with_logits(score, labels)
        reg_loss = self.regularization_loss(embed)
        return predict_loss + self.reg_param * reg_loss

def node_norm_to_edge_norm(g, node_norm):
    g = g.local_var()
    # convert to edge norm
    g.ndata['norm'] = node_norm
    g.apply_edges(lambda edges : {'norm' : edges.dst['norm']})
    return g.edata['norm']


"""Torch Module for Relational graph convolution layer"""

class RelGraphConvRefactor(dglModel.RelGraphConv):
    def __init__(self,
                 in_feat,
                 out_feat,
                 num_rels,
                 regularizer="nothing",
                 num_bases=None,
                 bias=True,
                 activation=None,
                 self_loop=False,
                 dropout=0.0,
                 reduce_func = fn.mean):
        super(RelGraphConvRefactor, self).__init__(in_feat, out_feat, num_rels, regularizer, num_bases
                                                   ,bias, activation, self_loop, dropout)

        self.reduce_func = reduce_func
        if regularizer == "basis":
            # add basis weights
            self.weight = nn.Parameter(torch.Tensor(self.num_bases, self.in_feat, self.out_feat))
            if self.num_bases < self.num_rels:
                # linear combination coefficients
                self.w_comp = nn.Parameter(torch.Tensor(self.num_rels, self.num_bases))
            nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))
            if self.num_bases < self.num_rels:
                nn.init.xavier_uniform_(self.w_comp,
                                        gain=nn.init.calculate_gain('relu'))
            # message func
            self.message_func = self.basis_message_func
        elif regularizer == "bdd":
            if in_feat % num_bases != 0 or out_feat % num_bases != 0:
                raise ValueError('Feature size must be a multiplier of num_bases.')
            # add block diagonal weights
            self.submat_in = in_feat // self.num_bases
            self.submat_out = out_feat // self.num_bases

            # assuming in_feat and out_feat are both divisible by num_bases
            self.weight = nn.Parameter(torch.Tensor(
                self.num_rels, self.num_bases * self.submat_in * self.submat_out))
            nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))
            # message func
            self.message_func = self.bdd_message_func

        elif regularizer == "nothing":
            self.message_func = self.nothing_message_func
        else:
            raise ValueError("Regularizer must be either 'basis' or 'bdd'")


    def nothing_message_func(self, edges):
        msg = edges.src['h']
        return {'msg' : msg}

    def forward(self, g, x, etypes, norm=None):
        g = g.local_var()
        g.ndata['h'] = x
        g.edata['type'] = etypes
        if norm is not None:
            g.edata['norm'] = norm
        if self.self_loop:
            loop_message = utils.matmul_maybe_select(x, self.loop_weight)
        # message passing
        g.update_all(self.message_func, self.reduce_func(msg='msg', out='h'))
        # apply bias and activation
        node_repr = g.ndata['h']
        if self.bias:
            node_repr = node_repr + self.h_bias
        if self.self_loop:
            node_repr = node_repr + loop_message
        if self.activation:
            node_repr = self.activation(node_repr)
        node_repr = self.dropout(node_repr)
        return node_repr
