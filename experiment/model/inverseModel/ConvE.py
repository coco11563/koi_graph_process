import torch
from torch.nn import functional as F, Parameter

from torch.nn.init import xavier_normal_


class ConvE(torch.nn.Module):
    def __init__(self,num_entities, num_relations, hidden_size = 13440 ,embedding_dim = 256, embedding_shape1 = 16, input_drop = 0.05, hidden_drop = 0.1, feat_drop = 0.05, use_bias = True):
        super(ConvE, self).__init__()
        self.emb_e = torch.nn.Embedding(num_entities, embedding_dim, padding_idx=0)
        self.emb_rel = torch.nn.Embedding(num_relations, embedding_dim, padding_idx=0)
        self.inp_drop = torch.nn.Dropout(input_drop)
        self.hidden_drop = torch.nn.Dropout(hidden_drop)
        self.feature_map_drop = torch.nn.Dropout2d(feat_drop)
        self.loss = torch.nn.BCELoss()
        self.emb_dim1 = embedding_shape1
        self.emb_dim2 = embedding_dim // self.emb_dim1

        self.conv1 = torch.nn.Conv2d(1, 32, (3, 3), 1, 0, bias=use_bias)
        self.bn0 = torch.nn.BatchNorm2d(1)
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.bn2 = torch.nn.BatchNorm1d(embedding_dim)
        self.register_parameter('b', Parameter(torch.zeros(num_entities)))
        self.fc = torch.nn.Linear(hidden_size, embedding_dim)
        print(num_entities, num_relations)

    def init(self):
        xavier_normal_(self.emb_e.weight.data)
        xavier_normal_(self.emb_rel.weight.data)

    def forward(self, e1, rel):
        e1_embedded= self.emb_e(e1).view(-1, 1, self.emb_dim1, self.emb_dim2)
        rel_embedded = self.emb_rel(rel).view(-1, 1, self.emb_dim1, self.emb_dim2)

        stacked_inputs = torch.cat([e1_embedded, rel_embedded], 2)

        stacked_inputs = self.bn0(stacked_inputs)
        x= self.inp_drop(stacked_inputs)
        x= self.conv1(x)
        x= self.bn1(x)
        x= F.relu(x)
        x = self.feature_map_drop(x)
        x = x.view(x.shape[0], -1)
        # print(x.shape)
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)
        # self.emb_e.weight.transpose(1,0) is all perhaps
        x = torch.mm(x, self.emb_e.weight.transpose(1,0))
        x += self.b.expand_as(x)
        pred = torch.sigmoid(x)

        return pred

def sort_and_rank(score, target):
    _, indices = torch.sort(score, dim=1, descending=True)
    indices = torch.nonzero(indices == target.view(-1, 1))
    indices = indices[:, 1].view(-1)
    return indices

def perturb_and_get_rank(model, a, r, b, test_size, batch_size=100):
    """ Perturb one element in the triplets
    """
    n_batch = (test_size + batch_size - 1) // batch_size
    ranks = []
    for idx in range(n_batch):
        # print("batch {} / {}".format(idx, n_batch))
        batch_start = idx * batch_size
        batch_end = min(test_size, (idx + 1) * batch_size)
        batch_a = a[batch_start: batch_end]
        batch_r = r[batch_start: batch_end]
        out = model(batch_a, batch_r)
        target = b[batch_start: batch_end]
        ranks.append(sort_and_rank(out, target))
    return torch.cat(ranks)

def calc_mrr(model, test_triplets, hits=[], eval_bz=100):
    with torch.no_grad():
        s = test_triplets[:, 0]
        r = test_triplets[:, 1]
        o = test_triplets[:, 2]
        test_size = test_triplets.shape[0]
        # w : w_relation in model
        # o : object
        # r : rel
        # s : subject
        # perturb subject
        # perturb_and_get_rank(embedding, w, a, r, b, test_size, batch_size=100):
        ranks_s = perturb_and_get_rank(model, o, r, s, test_size, eval_bz)
        # perturb object
        ranks_o = perturb_and_get_rank(model, s, r, o, test_size, eval_bz)

        ranks = torch.cat([ranks_s, ranks_o])
        ranks += 1 # change to 1-indexed

        mrr = torch.mean(1.0 / ranks.float())
        print("MRR (raw): {:.6f}".format(mrr.item()))

        for hit in hits:
            avg_count = torch.mean((ranks <= hit).float())
            print("Hits (raw) @ {}: {:.6f}".format(hit, avg_count.item()))
    return mrr.item()



