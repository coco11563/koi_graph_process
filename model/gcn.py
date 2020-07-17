import torch
import torch.nn as nn
import torch.nn.functional as F
from HeterGCN import heterrelgraphconv as heter
from HeterGCN import BaseHRGCN
import importlib
importlib.reload(BaseHRGCN)
importlib.reload(heter)
class EmbeddingLayer(nn.Module):
    def __init__(self, num_nodes, h_dim, feature_dcit = None, i2w_dict = None):
        super(EmbeddingLayer, self).__init__()
        self.embedding = torch.nn.Embedding(num_nodes, h_dim)
        if feature_dcit is not None and i2w_dict is not None :
            # init w2v
            for i in range(num_nodes) :
                if i2w_dict.__contains__(i) :
                    self.embedding[i] = torch.FloatTensor(feature_dcit[i2w_dict[i]])

    # squeeze remove useless dim

    def forward(self, g, h):
        return self.embedding(h.squeeze())


class HRGCN(BaseHRGCN.BaseHRGCN):

    def build_input_layer(self):
        return EmbeddingLayer(self.num_nodes, self.h_dim)

    def build_hidden_layer(self, idx):
        act = F.relu if idx < self.num_hidden_layers - 1 else None # 只给非最后一层hidden增加act fun
        return heter.HeterRelGraphConv(in_feat=self.h_dim, out_feat=self.h_dim, num_rels=self.num_rels, rel_dict=self.rel_dict, regularizer ="basis",
                num_bases= self.num_bases, activation=act, self_loop=True,
                dropout=self.dropout)


class LinkPredict(nn.Module):

    def __init__(self, in_dim, h_dim, num_rels, rel_dict, num_bases=-1,
                 num_hidden_layers=1, dropout=0, use_cuda=False, reg_param=0):
        super(LinkPredict, self).__init__()
        self.rgcn = HRGCN(in_dim, h_dim, h_dim, num_rels * 2, num_bases , rel_dict= rel_dict,
                         num_hidden_layers=num_hidden_layers, dropout=dropout, use_cuda=use_cuda)
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

    def forward(self, g, h):
        return self.rgcn.forward(g, h)

    def regularization_loss(self, embedding):
        return torch.mean(embedding.pow(2)) + torch.mean(self.w_relation.pow(2))

    def get_loss(self, g, embed, triplets, labels):
        # triplets is a list of data samples (positive and negative)
        # each row in the triplets is a 3-tuple of (source, relation, destination)
        score = self.calc_score(embed, triplets)
        predict_loss = F.binary_cross_entropy_with_logits(score, labels)
        reg_loss = self.regularization_loss(embed) # 正则项
        return predict_loss + self.reg_param * reg_loss