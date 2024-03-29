# -*- coding: utf-8 -*-
# @Filename: GTransformer_graseq.py
# @Date: 2022-09-02 08:48
# @Author: Leo Xu
# @Email: leoxc1571@163.com
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
    def __init__(self, hidden_dim, eps=1e-12):
        super(BertLayerNorm, self).__init__()
        self.shape = torch.Size((hidden_dim,))
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_dim))
        self.bias = nn.Parameter(torch.zeros(hidden_dim))
        self.apex_enabled = APEX_IS_AVAILABLE

    @torch.jit.unused
    def fused_layer_norm(self, x):
        return FusedLayerNormAffineFunction.apply(
            x, self.weight, self.bias, self.shape, self.eps
        )

    def forward(self, x):
        if self.apex_enabled and not torch.jit.is_scripting():
            x = self.fused_layer_norm(x)
        else:
            u = x.mean(-1, keepdim=True)
            s = (x - u).pow(2).mean(-1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight * x + self.bias
        return x


def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / 1.41421))


def bias_gelu(bias, y):
    x = bias + y
    return x * 0.5 * (1.0 + torch.erf(x / 1.41421))


class LinearActivation(nn.Module):
    def __init__(self, in_features, out_features,  bias=True):
        super(LinearActivation, self).__init__()
        self.in_features = in_features
        self.out_features = out_features                                                                 #
        if bias:  # compatibility
            self.biased_act_fn =bias_gelu
        else:
            self.act_fn = gelu
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        if not self.bias is None:
            return self.biased_act_fn(self.bias, F.linear(input, self.weight, None))
        else:
            return self.act_fn(F.linear(input, self.weight, self.bias))

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )



class GTOut(nn.Module):
    def __init__(self, hidden_dim, drop_ratio):
        super(GTOut, self).__init__()
        self.dense = nn.Linear(4 * hidden_dim, hidden_dim)
        self.LayerNorm = BertLayerNorm(hidden_dim, eps=1e-12)
        self.dropout = nn.Dropout(drop_ratio)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class Intermediate(nn.Module):
    def __init__(self, hidden_dim):
        super(Intermediate, self).__init__()
        self.dense_act = LinearActivation(hidden_dim, 4 * hidden_dim)

    def forward(self, hidden_states):
        hidden_states = self.dense_act(hidden_states)
        return hidden_states


class AttentionOut(nn.Module):
    def __init__(self, hidden_dim, drop_ratio=0.):
        super(AttentionOut, self).__init__()
        self.dense = nn.Linear(hidden_dim, hidden_dim)
        self.LayerNorm = BertLayerNorm(hidden_dim, eps=1e-12)
        self.dropout = nn.Dropout(drop_ratio)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class GraphAttentionConv(MessagePassing):
    def __init__(self, hidden_dim, heads=3, dropout=0.):
        super(GraphAttentionConv, self).__init__()
        assert hidden_dim % heads == 0
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.node_dim = 0
        self.attention_drop = nn.Dropout(dropout)

        self.query = nn.Linear(hidden_dim, heads * int(hidden_dim / heads))
        self.key = nn.Linear(hidden_dim, heads * int(hidden_dim / heads))
        self.value = nn.Linear(hidden_dim, heads * int(hidden_dim / heads))

        self.init_params()

    def init_params(self):
        nn.init.xavier_uniform_(self.query.weight.data)
        nn.init.xavier_uniform_(self.key.weight.data)
        nn.init.xavier_uniform_(self.value.weight.data)

    def forward(self, x, edge_index, edge_attr, size=None):
        pseudo = edge_attr.unsqueeze(-1) if edge_attr.dim() == 1 else edge_attr
        return self.propagate(edge_index=edge_index, size=size, x=x, pseudo=pseudo)

    def message(self, edge_index_i, x_i, x_j, pseudo, size_i):
        query = self.query(x_i).view(-1, self.heads, int(self.hidden_dim / self.heads))
        key = self.key(x_j + pseudo).view(-1, self.heads, int(self.hidden_dim / self.heads))
        value = self.value(x_j + pseudo).view(-1, self.heads, int(self.hidden_dim / self.heads))

        alpha = (query * key).sum(dim=-1) / math.sqrt(int(self.hidden_dim / self.heads))
        alpha = softmax(alpha, edge_index_i, num_nodes=size_i)
        alpha = self.attention_drop(alpha.view(-1, self.heads, 1))

        return alpha * value

    def update(self, aggr_out):
        aggr_out = aggr_out.view(-1, self.heads * int(self.hidden_dim / self.heads))
        return aggr_out


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
    def __init__(self, num_layer, emb_dim, heads, num_message_passing, num_tasks,
                 drop_ratio=0, graph_pooling='mean', device='cpu', fusion=True,
                 num_atom_feat=178, num_bond_feat=18):
        super(MolGNet, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.num_tasks = num_tasks
        self.emb_dim = emb_dim
        self.device = device
        self.fusion = fusion
        self.x_embedding = nn.Embedding(num_atom_feat, emb_dim).to(self.device)
        self.x_seg_embedding = nn.Embedding(seg_size, emb_dim).to(self.device)
        self.edge_embedding = nn.Embedding(num_bond_feat, emb_dim).to(self.device)
        self.edge_seg_embedding = nn.Embedding(seg_size, emb_dim).to(self.device)
        self.x_embedding = nn.Embedding(num_atom_feat, emb_dim).to(self.device)
        self.x_seg_embedding = nn.Embedding(seg_size, emb_dim).to(self.device)
        # self.smile_embedding = nn.Embedding(178, 256).to(self.device)
        # self.smile_seg_embedding = nn.Embedding(seg_size, 300).to(self.device)

        if self.num_layer < 2:
            raise ValueError('Number of GNN layers must be greater than 1.')
        self.dummy = False
        self.mult = 1
        self.hidden_size = 128

        self.init_params()

        self.LSTM = nn.LSTM(input_size=emb_dim, hidden_size=self.hidden_size,
                            bidirectional=True, num_layers=2).to(self.device)
        self.gnns = nn.ModuleList([GTLayer(emb_dim, heads, num_message_passing, drop_ratio)
                                   for _ in range(num_layer)]).to(self.device)
        self.graph_reduce_linear = nn.Linear(self.mult * self.emb_dim, self.hidden_size * 2).to(self.device)
        self.graph_pred_linear = nn.Linear(self.hidden_size * 2, self.num_tasks).to(self.device)
        self.fusion_pred = nn.Sequential(nn.Linear(self.hidden_size * 2, 16),
                                         nn.ReLU(),
                                         nn.Linear(16, num_tasks)).to(self.device)
        # self.lstm_pred_linear = nn.Linear(self.hidden_size, self.num_tasks).to(self.device)

        if graph_pooling == 'sum':
            self.pool = global_add_pool
        elif graph_pooling == 'mean':
            self.pool = global_mean_pool
        elif graph_pooling == 'max':
            self.pool = global_max_pool
        elif graph_pooling == 'attention':
            self.pool = GlobalAttention(gate_nn=nn.Linear(emb_dim, 1))
        elif graph_pooling == 'set2set':
            self.pool = Set2Set(emb_dim, 3)
            self.mult = 2
        elif graph_pooling == 'collection':
            self.dummy = True
        else:
            raise ValueError('Invalid graph pooling type.')

    def init_params(self):
        nn.init.xavier_uniform_(self.x_embedding.weight.data)
        nn.init.xavier_uniform_(self.x_seg_embedding.weight.data)
        nn.init.xavier_uniform_(self.edge_embedding.weight.data)
        nn.init.xavier_uniform_(self.edge_seg_embedding.weight.data)
        # nn.init.xavier_uniform_(self.smile_embedding.weight.data)
        # nn.init.xavier_uniform_(self.smile_seg_embedding.weight.data)

    def split_tensor(self, x: torch.Tensor, batch: torch.Tensor):
        length = torch.unique(batch, return_counts=True)[1].cpu()
        dup_count = tuple(length.numpy())
        split = x.split(dup_count, dim=0)
        return list(split), length

    def forward(self, *argv):
        if len(argv) == 6:
            x, edge_idx, edge_attr, batch, node_seg, edge_seg, dummy_indice = \
                argv[0], argv[1], argv[2], argv[3], argv[4], argv[5], argv[6]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_idx, edge_attr, batch, node_seg, edge_seg, dummy_indice = \
                data.x, data.edge_index, data.edge_attr, data.batch, data.node_seg, \
                data.edge_seg, data.dummy_node_indices
        else:
            raise ValueError('Unmatched number of arguments')

        # x_seq = x
        x = self.x_embedding(x).sum(1) + self.x_seg_embedding(node_seg)
        # x_seq = self.smile_embedding(x_seq).sum(1)
        edge_attr = self.edge_embedding(edge_attr).sum(1) + self.edge_seg_embedding(edge_seg)

        for gnn in self.gnns:
            x = gnn(x, edge_idx, edge_attr)

        split_x, length = self.split_tensor(x, batch)
        pad_x = nn.utils.rnn.pad_sequence(split_x, batch_first=True)
        packed_x = nn.utils.rnn.pack_padded_sequence(pad_x, lengths=length, batch_first=True, enforce_sorted=False)
        lstm, _ = self.LSTM(packed_x)
        unpack_x, _ = nn.utils.rnn.pad_packed_sequence(lstm, batch_first=True)
        unpack_x = torch.sum(unpack_x, dim=1)



        node_representation = x
        if self.fusion:
            mol_emb = F.normalize(self.graph_reduce_linear(self.pool(node_representation, batch)), p=2, dim=0) + \
                      F.normalize(unpack_x, p=2, dim=0)
            return self.fusion_pred(mol_emb)
        else:
            if self.dummy:
                return self.graph_pred_linear(node_representation[dummy_indice])
            else:
                return self.graph_pred_linear(torch.concat((self.graph_reduce_linear(self.pool(node_representation, batch)), unpack_x), dim=1))
