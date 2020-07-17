import dgl.function as fn
import torch.nn as nn
import torch as torch
import torch.nn.functional as F

class HeteroRGCNLayer(nn.Module):
    def __init__(self, in_size, out_size, etypes):
        super(HeteroRGCNLayer, self).__init__()
        # W_r for each relation 为每一种不同的关系赋予一个不同的权值矩阵
        self.weight = nn.ModuleDict({
                name : nn.Linear(in_size, out_size) for name in etypes
            })

    def forward(self, G, feat_dict):
        # 输入一个DGL图 一个feature dict
        # The input is a dictionary of node features for each type
        funcs = {}
        for srctype, etype, dsttype in G.canonical_etypes:
            # g.etype : ['written-by', 'writing', 'citing', 'cited', 'is-about', 'has']
            #  G.canonical_etypes : [('paper', 'written-by', 'author'),
            # ('author', 'writing', 'paper'),
            # ('paper', 'citing', 'paper'),
            # ('paper', 'cited', 'paper'),
            # ('paper', 'is-about', 'subject'),
            # ('subject', 'has', 'paper')]
            # Compute W_r * h
            # weight schema is [name : nn linear]
            Wh = self.weight[etype](feat_dict[srctype])
            # Save it in graph for message passing
            G.nodes[srctype].data['Wh_%s' % etype] = Wh
            # Specify per-relation message passing functions: (message_func, reduce_func).
            # Note that the results are saved to the same destination feature 'h', which
            # hints the type wise reducer for aggregation.
            funcs[etype] = (fn.copy_u('Wh_%s' % etype, 'm'), fn.mean('m', 'h'))
        # Trigger message passing of multiple types.
        # The first argument is the message passing functions for each relation.
        # The second one is the type wise reducer, could be "sum", "max",
        # "min", "mean", "stack"
        G.multi_update_all(funcs, 'sum')
        # return the updated node feature dictionary
        return {ntype : G.nodes[ntype].data['h'] for ntype in G.ntypes}

class HeteroRGCN(nn.Module):
    def __init__(self, G, in_size, hidden_size, out_size):
        super(HeteroRGCN, self).__init__()
        # Use trainable node embeddings as featureless inputs.
        embed_dict = {ntype : nn.Parameter(torch.Tensor(G.number_of_nodes(ntype), in_size))
                      for ntype in G.ntypes}
        for key, embed in embed_dict.items():
            nn.init.xavier_uniform_(embed)
        self.embed = nn.ParameterDict(embed_dict)
        # create layers
        self.layer1 = HeteroRGCNLayer(in_size, hidden_size, G.etypes)
        self.layer2 = HeteroRGCNLayer(hidden_size, out_size, G.etypes)

    def forward(self, G):
        h_dict = self.layer1(G, self.embed)
        h_dict = {k : F.leaky_relu(h) for k, h in h_dict.items()}
        h_dict = self.layer2(G, h_dict)
        # get paper logits
        return h_dict['paper']

class NormalGCNLayer(nn.Module) :
    def __init__(self, in_size, out_size):
        super(NormalGCNLayer, self).__init__()
        self.weight = nn.Linear(in_size, out_size)

    def forward(self, G, feat_dict):
        funcs = {}
        for srctype, etype, dsttype in G.canonical_etypes:
            Wh = self.weight(feat_dict[srctype])
            G.nodes[srctype].data['Wh'] = Wh
            funcs[etype] = (fn.copy_u('Wh', 'm'), fn.mean('m', 'h'))
        G.multi_update_all(funcs,'sum')
        return {ntype : G.nodes[ntype].data['h'] for ntype in G.ntypes}

class NormalGCN(nn.Module):
    def __init__(self, G, in_size, hidden_size, out_size):
        super(NormalGCN, self).__init__()
        # Use trainable node embeddings as featureless inputs.
        embed_dict = {ntype: nn.Parameter(torch.Tensor(G.number_of_nodes(ntype), in_size))
                      for ntype in G.ntypes}
        for key, embed in embed_dict.items():
            nn.init.xavier_uniform_(embed)
            print(embed.device)
        self.embed = embed_dict
        # create layers
        self.layer1 = NormalGCNLayer(in_size, hidden_size)
        self.layer2 = NormalGCNLayer(hidden_size, out_size)

    def forward(self, G):
        h_dict = self.layer1(G, self.embed)
        h_dict = {k : F.leaky_relu(h) for k, h in h_dict.items()}
        h_dict = self.layer2(G, h_dict)
        # get paper logits
        return h_dict
# Sends a message of node feature h.
msg = fn.copy_src(src='h', out='m')

def reduce(nodes):
    """Take an average over all neighbor node features hu and use it to
    overwrite the original node feature."""
    accum = torch.mean(nodes.mailbox['m'], 1)
    return {'h': accum}

class NodeApplyModule(nn.Module):
    """Update the node feature hv with ReLU(Whv+b)."""
    def __init__(self, in_feats, out_feats, activation):
        super(NodeApplyModule, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        self.activation = activation

    def forward(self, node):
        h = self.linear(node.data['h'])
        h = self.activation(h)
        return {'h' : h}

class GCN(nn.Module):
    def __init__(self, in_feats, out_feats, activation):
        super(GCN, self).__init__()
        self.apply_mod = NodeApplyModule(in_feats, out_feats, activation)

    def forward(self, g, feature):
        # Initialize the node features with h.
        g.ndata['h'] = feature
        g.update_all(msg, reduce)
        g.apply_nodes(func=self.apply_mod)
        return g.ndata.pop('h')