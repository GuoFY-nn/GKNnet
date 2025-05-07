import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
import torch.nn as nn
import torch.nn.functional as F
from kan.KANLayer import *
from torchcubicspline import NaturalCubicSpline

class RGCNLayer(MessagePassing):
    def __init__(self, in_channels, out_channels, num_relations):
        super(RGCNLayer, self).__init__(aggr='add')
        self.num_relations = num_relations
        self.linear = nn.Linear(in_channels, out_channels)
        self.relation_weights = nn.ParameterList([
            nn.Parameter(torch.Tensor(in_channels, out_channels)) for _ in range(num_relations)
        ])
        self.reset_parameters()

    def reset_parameters(self):
        for weight in self.relation_weights:
            nn.init.xavier_uniform_(weight)
        nn.init.xavier_uniform_(self.linear.weight)
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)

    def forward(self, x, edge_indices):
        out = torch.zeros(x.size(0), self.linear.out_features, device=x.device)
        for i, edge_index in enumerate(edge_indices):
            if edge_index.numel() > 0:
                edge_weight = torch.ones(edge_index.size(1), device=x.device)
                adj_t = torch.sparse_coo_tensor(edge_index, edge_weight, (x.size(0), x.size(0)))
                out += torch.sparse.mm(adj_t, x @ self.relation_weights[i])
        out = self.linear(out)
        return out
class SplineActivation(nn.Module):
    def __init__(self, out_features, num_knots=5):
        super(SplineActivation, self).__init__()
        self.knots = nn.Parameter(torch.linspace(-1, 1, num_knots))
        self.values = nn.Parameter(torch.randn(out_features, num_knots))
        self.spline = NaturalCubicSpline(self.knots, self.values)

    def forward(self, x):
        return self.spline.evaluate(x)

class KANLayer(nn.Module):
    def __init__(self, in_feat, out_feat, grid_feat, addbias=True):
        super(KANLayer, self).__init__()
        self.lin_in = nn.Linear(in_feat, out_feat, bias=addbias)
        self.alpha = nn.Parameter(torch.ones(out_feat))
        self.beta = nn.Parameter(torch.zeros(out_feat))
        self.spline_act = SplineActivation(out_feat, grid_feat)

    def forward(self, x):
        x = self.lin_in(x)
        x = self.alpha * x + self.beta
        x = self.spline_act(x)
        return x

class GKNnet(nn.Module):
    def __init__(self, input_features, hidden_features, output_classes, num_relations=3, use_bias=True):
        super(GKNnet, self).__init__()

        self.rgcn1 = RGCNLayer(input_features, hidden_features, num_relations)
        self.rgcn2 = RGCNLayer(hidden_features, hidden_features, num_relations)


        self.kan_layer = KANLayer(hidden_features, hidden_features, grid_feat, addbias=use_bias)

        self.fc = nn.Linear(hidden_features, output_classes)

    def forward(self, data):
        x, edge_indices = data.x, [data.edge_index_A1, data.edge_index_A2, data.edge_index_A3]
        x = self.rgcn1(x, edge_indices)
        x = F.relu(x)
        x = self.rgcn2(x, edge_indices)
        x = F.relu(x)
        x = self.kan_layer(x)
        x = self.fc(x)
        return x.squeeze(1)
