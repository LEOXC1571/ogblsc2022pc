from turtle import forward
import torch.nn as nn
import torch
from torch_geometric.nn import MessagePassing
from .comenet import dist_calc, angle_emb, torsion_emb
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool

### GIN convolution along the graph structure
class GINConv(MessagePassing):
    def __init__(self, emb_dim):
        '''
            emb_dim (int): node embedding dimensionality
        '''

        super(GINConv, self).__init__(aggr="add")

        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, emb_dim), torch.nn.BatchNorm1d(emb_dim),
                                       torch.nn.ReLU(), torch.nn.Linear(emb_dim, emb_dim))
        self.eps = torch.nn.Parameter(torch.Tensor([0]))


    def forward(self, x, edge_index, edge_attr):
        out = self.mlp((1 + self.eps) * x + self.propagate(edge_index, x=x, edge_attr=edge_attr))

        return out

    def message(self, x_j, edge_attr):
        return F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out

class SimComE(nn.Module):
    def __init__(self,
                cutoff=8.0,
                num_layers=4,
                hidden_channels=300,
                out_channels=1,
                num_radial=3,
                num_spherical=2,
                residual=False):
        super(SimComE, self).__init__()
        self.cutoff = cutoff
        self.num_layers = num_layers
        self.residual = residual
        
        self.x_emb = nn.Embedding(178, hidden_channels)
        self.edge_emb = nn.Embedding(18, hidden_channels)

        self.feat1 = torsion_emb(num_radial=num_radial, num_spherical=num_spherical, cutoff=cutoff)
        self.feat2 = angle_emb(num_radial=num_radial, num_spherical=num_spherical, cutoff=cutoff)

        self.lin_feat1 =nn.Linear(num_radial * num_spherical ** 2, hidden_channels)
        self.lin_feat2 = nn.Linear(num_radial * num_spherical, hidden_channels)
        self.linear_cat = nn.Linear(3 * hidden_channels, hidden_channels)

        ### set the initial virtual node embedding to 0.
        self.virtualnode_embedding = torch.nn.Embedding(1, hidden_channels)
        torch.nn.init.constant_(self.virtualnode_embedding.weight.data, 0)

        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        ### List of MLPs to transform virtual node at every layer
        self.mlp_virtualnode_list = torch.nn.ModuleList()

        for _ in range(num_layers):
            self.convs.append(GINConv(hidden_channels))
            self.batch_norms.append(nn.BatchNorm1d(hidden_channels))

        for _ in range(num_layers - 1):
            self.mlp_virtualnode_list.append(
                torch.nn.Sequential(torch.nn.Linear(hidden_channels, hidden_channels), torch.nn.BatchNorm1d(hidden_channels), torch.nn.ReLU(),
                                    torch.nn.Linear(hidden_channels, hidden_channels), torch.nn.BatchNorm1d(hidden_channels), torch.nn.ReLU()))

        self.graph_pred_linear = nn.Linear(hidden_channels, out_channels)

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.x_emb.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_emb.weight.data)
        torch.nn.init.xavier_uniform_(self.feat1.weight.data)
        torch.nn.init.xavier_uniform_(self.feat2.weight.data)

    def forward(self, data):
        x, edge_index, edge_attr, pos, batch = data.x, data.edge_index, data.edge_attr, \
            data.pos, data.batch


        # 3D information
        num_nodes = x.size(0)
        i, j = edge_index[0], edge_index[1]
        vecs = pos[j] - pos[i]
        dist = vecs.norm(dim=-1)
        theta, phi, tau = dist_calc(self.cutoff, vecs, dist, i, j, num_nodes)

        # Embeddings
        x = self.x_emb(x).sum(1)
        edge_attr = self.edge_emb(edge_attr).sum(1)
        feat1 = self.feat1(dist, theta, phi)
        feat2 = self.feat2(dist, tau)
        feat1 = self.lin_feat1(feat1)
        feat2 = self.lin_feat2(feat2)
        # num_e * 3 * hidden channels to hidden
        edge_attr_cat = self.linear_cat(torch.cat([edge_attr, feat1, feat2], dim=1))

        ### virtual node embeddings for graphs
        virtualnode_embedding = self.virtualnode_embedding(
            torch.zeros(batch[-1].item() + 1).to(edge_index.dtype).to(edge_index.device))

        h_list = [x]
        for layer in range(self.num_layers):
            h_list[layer] = h_list[layer] + virtualnode_embedding[batch]

            ### Message passing among graph nodes
            h = self.convs[layer](h_list[layer], edge_index, edge_attr_cat)
            h = self.batch_norms[layer](h)
            h = F.relu(h)

            if self.residual:
                h = h + h_list[layer]

            h_list.append(h)

            ### update the virtual nodes
            if layer < self.num_layers - 1:
                ### add message from graph nodes to virtual nodes
                virtualnode_embedding_temp = global_add_pool(h_list[layer], batch) + virtualnode_embedding
                ### transform virtual nodes using MLP

                if self.residual:
                    virtualnode_embedding = virtualnode_embedding + \
                        self.mlp_virtualnode_list[layer](virtualnode_embedding_temp)
                else:
                    virtualnode_embedding = self.mlp_virtualnode_list[layer](virtualnode_embedding_temp)

        h_node = h_list[-1]
        h_graph = global_add_pool(h_node, batch)
        output = self.graph_pred_linear(h_graph)
        return output