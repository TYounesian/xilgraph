from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool


class GCN(nn.Module):
    def __init__(self, in_dim=7, hidden=16, out_dim=2, dropout=0):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.conv3 = GCNConv(hidden, hidden)
        self.dropout = nn.Dropout(dropout)
        self.lin = nn.Linear(hidden, out_dim)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv3(x, edge_index))
        x = self.dropout(x)
        x = global_mean_pool(x, batch)
        return self.lin(x)


class GAT(nn.Module):
    def __init__(self, in_dim=7,  hidden=16, heads1=4, heads2=4, out_dim=2, attn_dropout=0.2, feat_dropout=0.2):
        super().__init__()
        self.gat1 = GATConv(
            in_channels=in_dim,
            out_channels=hidden,
            heads=heads1,
            concat=True,
            dropout=attn_dropout,
        )
        self.gat2 = GATConv(
            in_channels=hidden * heads1,
            out_channels=hidden,
            heads=heads2,
            concat=False,
            dropout=attn_dropout,
        )

        self.bn1 = nn.BatchNorm1d(hidden * heads1)
        self.bn2 = nn.BatchNorm1d(hidden)

        self.feat_dropout = nn.Dropout(feat_dropout)
        self.pool = global_mean_pool
        self.lin = nn.Linear(hidden, out_dim)

    def forward(self, x, edge_index, batch):
        x = F.elu(self.bn1(self.gat1(x, edge_index)))
        x = self.feat_dropout(x)

        x = F.elu(self.bn2(self.gat2(x, edge_index)))
        x = self.feat_dropout(x)

        x = self.pool(x, batch)
        return self.lin(x)