# Coding: utf-8
# Coded By Leo Xu
# At 2022/7/20 16:11
# Email: leoxc1571@163.com

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add

from torch_geometric.nn import MessagePassing, global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, \
    Set2Set
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import add_self_loops, degree, softmax

num_atom_type = 120 + 1 + 1
num_chirality_tag = 3

num_bond_type = 6 + 1 + 1
num_bond_direction = 3

seg_size = 3

try:
    import apex
    # apex.amp.register_half_function(apex.normalization.fused_layer_norm, 'FusedLayerNorm')
    import apex.normalization
    from apex.normalization.fused_layer_norm import FusedLayerNormAffineFunction
    # apex.amp.register_float_function(apex.normalization.FusedLayerNorm, 'forward')
    # BertLayerNorm = apex.normalization.FusedLayerNorm
    APEX_IS_AVAILABLE = True
except ImportError:
    print("Better speed can be achieved with apex installed from https://www.github.com/nvidia/apex.")
    # BertLayerNorm = BertNonFusedLayerNorm
    APEX_IS_AVAILABLE = False


class BertLayerNorm(nn.Module):
    def __init__(self, hidden_dim, eps):
        super(BertLayerNorm, self).__init__()


class GTOut(nn.Module):
    def __init__(self, hidden_dim, drop_ratio):
        super(GTOut, self).__init__()


class Intermediate(nn.Module):
    def __init__(self, hidden_dim):
        super(Intermediate, self).__init__()


class AttentionOut(nn.Module):
    def __init__(self, hidden_dim, drop_ratio=0.):
        super(AttentionOut, self).__init__()


class GraphAttentionConv(MessagePassing):
    def __init__(self, hidden_dim, heads=3, dropout=0.):
        super(GraphAttentionConv, self).__init__()


class GTLayer(nn.Module):
    def __init__(self, hidden_dim, heads, num_message_passing, drop_ratio):
        super(GTLayer, self).__init__()
        self.attention = GraphAttentionConv(hidden_dim, heads, drop_ratio)
        self.attentionout = AttentionOut(hidden_dim, drop_ratio)
        self.intermediate = Intermediate(hidden_dim)
        self.output = GTOut(hidden_dim, drop_ratio)
        self.gru = nn.GRU(hidden_dim, hidden_dim)
        self.layernorm = BertLayerNorm(hidden_dim, eps=1e-12)
        self.time_step = num_message_passing

    def forward(self, x, edge_idx, edge_attr):
        h = x.unsqueeze(0)
        for i in range(self.time_step):
            attention_out = self.attention.forward(x, edge_idx, edge_attr)
            attention_out = self.attentionout.forward(attention_out, x)
            intermediate_out = self.intermediate.forward(attention_out)
            m = self.output.forward(intermediate_out, attention_out)
            x, h = self.gru(m.unsqueeze(0), h)
            x = self.layernorm.forward(x.squeeze(0))
        return x


class MolGNet(nn.Module):
    def __init__(self, num_layer, emb_dim, heads, num_message_passing, drop_ratio=0):
        super(MolGNet, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.x_embedding = nn.Embedding(178, emb_dim)
        self.x_seg_embedding = nn.Embedding(seg_size, emb_dim)
        self.edge_embedding = nn.Embedding(18, emb_dim)
        self.edge_seg_embedding = nn.Embedding(seg_size, emb_dim)

        self.init_params()

        self.gnn = nn.ModuleList([GTLayer(emb_dim, heads, num_message_passing, drop_ratio) for _ in range(num_layer)])

    def init_params(self):
        nn.init.xavier_uniform_(self.x_embedding.weight.data)
        nn.init.xavier_uniform_(self.x_seg_embedding.weight.data)
        nn.init.xavier_uniform_(self.edge_embedding.weight.data)
        nn.init.xavier_uniform_(self.edge_seg_embedding.weight.data)

    def forward(self, *argv):
        if len(argv) == 5:
            x, edge_idx, edge_attr, node_seg, edge_seg = argv[0], argv[1], argv[2], argv[3], argv[4]
        elif len(argv) == 1:
            data = argv[0]

