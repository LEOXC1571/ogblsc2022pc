from .comenet import dist_calc, angle_emb, torsion_emb
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GraphConv, GraphNorm


class EdgeGraphConv(GraphConv):
    def message(self, x_j, edge_weight):
        return x_j if edge_weight is None else edge_weight * x_j


class InterBock(nn.Module):
    def __init__(self,
                hidden_channels,
                middle_channels,
                num_radial,
                num_spherical,
                num_layers,
                output_channels):
        super(InterBock, self).__init__()
    
        # featx mlp
        self.lin_x = nn.Sequential(nn.Linear(hidden_channels, hidden_channels), \
            torch.nn.BatchNorm1d(hidden_channels), nn.ReLU())
        
        # feat1 to hidden_channels
        self.lin_feat1 = nn.Sequential(nn.Linear(num_radial * num_spherical ** 2, middle_channels), \
            torch.nn.BatchNorm1d(hidden_channels), nn.ReLU(), nn.Linear(middle_channels, hidden_channels),\
                nn.ReLU())
        # feat2 to hidden_channels
        self.lin_feat2 = nn.Sequential(nn.Linear(num_radial * num_spherical, middle_channels), \
            torch.nn.BatchNorm1d(hidden_channels), nn.ReLU(), nn.Linear(middle_channels, hidden_channels), \
                nn.ReLU())

        # feat1 local conv
        self.conv1 = EdgeGraphConv(hidden_channels, hidden_channels)
        # feat2 local conv
        self.conv2 = EdgeGraphConv(hidden_channels, hidden_channels)

        self.lin_cat = nn.Linear(2 * hidden_channels, hidden_channels)
        self.linears = nn.ModuleList()
        for _ in range(num_layers):
            self.linears.append(nn.Linear(hidden_channels, hidden_channels))
        self.norm = GraphNorm(hidden_channels)
        self.final = nn.Linear(hidden_channels, output_channels)


    def forward(self, x, edge_index, edge_attr, feature1, feature2, batch):
        x = self.lin_x(x)

        h1 = self.lin_feat1(feature1)
        h1 = self.conv1(x, edge_index, feature1)
        h1 = F.relu(h1)

        h2 = self.lin_feat2(feature2)
        h2 = self.conv1(x, edge_index, feature2)
        h2 = F.relu(h2)

        # cat & down-project
        h = F.relu(self.linear_cat(torch.cat([h1, h2], dim=1)))

        h = h + x
        for lin in self.linears:
            h = self.act(lin(h)) + h
        h = self.norm(h, batch)
        h = self.final(h)
        # num_out_layers mlp
        return h


class SimComE(nn.Module):
    def __init__(self,
                cutoff=8.0,
                num_layers=4,
                hidden_channels=256,
                middle_channels=64,
                out_channels=1,
                num_radial=3,
                num_spherical=2,
                num_output_layers=3):
        super(SimComE, self).__init__()
        self.cutoff = cutoff
        self.num_layers = num_layers
        self.out_channels = out_channels

        self.x_emb = nn.Embedding(178, hidden_channels)
        self.edge_emb = nn.Embedding(18, hidden_channels)

        self.feat1 = torsion_emb(num_radial=num_radial, num_spherical=num_spherical, cutoff=cutoff)
        self.feat2 = angle_emb(num_radial=num_radial, num_spherical=num_spherical, cutoff=cutoff)

        self.inter_block = nn.ModuleList(
            [
                InterBock(
                    hidden_channels,
                    middle_channels,
                    num_radial,
                    num_spherical,
                    num_output_layers,
                    hidden_channels,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, *argv):
        data = argv[0]
        x, edge_index, edge_attr, pos, batch = data.x, data.edge_index, data.edge_attr, \
            data.pos, data.batch

        # Embedding
        x = self.x_emb(x).sum(1)
        edge_attr = self.edge_emb(edge_attr).sum(1)

        # 3D information
        num_nodes = x.size(0)
        i, j = edge_index[0], edge_index[1]
        vecs = pos[j] - pos[i]
        dist = vecs.norm(dim=-1)
        dist, theta, phi, tau = dist_calc(self.cutoff, vecs, dist, i, j, num_nodes)

        # 3D info to vectors
        feat1 = self.feat1(dist, theta, phi)
        feat2 = self.feat2(dist, tau)

        for inter in self.inter_block:
            x = inter(x, edge_index, edge_attr, feat1, feat2, batch)

        