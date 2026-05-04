import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GCNConv, GATConv, GATv2Conv, GINConv, global_mean_pool, global_add_pool, global_max_pool


# SEED = 42
# torch.manual_seed(SEED)
class GCN(nn.Module):
    def __init__(self, in_dim=7, hidden=128, out_dim=2, dropout=0.3):
        super().__init__()

        self.conv1 = GCNConv(in_dim, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.conv3 = GCNConv(hidden, hidden)

        self.bn1 = nn.BatchNorm1d(hidden)
        self.bn2 = nn.BatchNorm1d(hidden)
        self.bn3 = nn.BatchNorm1d(hidden)

        self.dropout = nn.Dropout(dropout)

        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.relu(x)

        x = global_add_pool(x, batch)
        return self.head(x)

class SAGE(nn.Module):
    def __init__(self, in_dim=7, hidden=128, out_dim=2, dropout=0):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden)
        self.conv2 = SAGEConv(hidden, hidden)
        self.conv3 = SAGEConv(hidden, hidden)
        self.conv4 = SAGEConv(hidden, hidden)
        self.conv5 = SAGEConv(hidden, hidden)
        self.dropout = nn.Dropout(dropout)
        self.lin = nn.Linear(hidden, out_dim)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv3(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv4(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv5(x, edge_index))
        x = self.dropout(x)
        x = global_add_pool(x, batch)
        return self.lin(x)


class GAT(nn.Module):
    def __init__(self, in_dim=7, hidden=128, heads1=4, heads2=4, out_dim=2, attn_dropout=0.0, feat_dropout=0.0):
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
            concat=True,
            dropout=attn_dropout,
        )
        self.gat3 = GATConv(
            in_channels=hidden * heads2,
            out_channels=hidden,
            heads=heads2,
            concat=False,
            dropout=attn_dropout,
        )

        # self.bn1 = nn.BatchNorm1d(hidden * heads1)
        # self.bn2 = nn.BatchNorm1d(hidden)
        # self.bn3 = nn.BatchNorm1d(hidden)

        self.feat_dropout = nn.Dropout(feat_dropout)
        self.pool = global_add_pool
        self.lin = nn.Linear(hidden, out_dim)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.gat1(x, edge_index))
        x = self.feat_dropout(x)

        x = F.relu(self.gat2(x, edge_index))
        x = self.feat_dropout(x)

        x = F.relu(self.gat3(x, edge_index))
        x = self.feat_dropout(x)

        x = self.pool(x, batch)
        return self.lin(x)


class GATv2(nn.Module):
    def __init__(self, in_dim=7, hidden=100, heads1=4, heads2=4, out_dim=2, attn_dropout=0.0, feat_dropout=0.0):
        super().__init__()
        self.gat1 = GATv2Conv(
            in_channels=in_dim,
            out_channels=hidden,
            heads=heads1,
            concat=True,
            dropout=attn_dropout,
        )
        self.gat2 = GATv2Conv(
            in_channels=hidden * heads1,
            out_channels=hidden,
            heads=heads2,
            concat=True,
            dropout=attn_dropout,
        )
        self.gat3 = GATv2Conv(
            in_channels=hidden * heads2,
            out_channels=hidden,
            heads=heads2,
            concat=False,
            dropout=attn_dropout,
        )

        # self.bn1 = nn.BatchNorm1d(hidden * heads1)
        # self.bn2 = nn.BatchNorm1d(hidden)
        # self.bn3 = nn.BatchNorm1d(hidden)

        self.feat_dropout = nn.Dropout(feat_dropout)
        self.pool = global_mean_pool
        self.lin = nn.Linear(hidden, out_dim)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.gat1(x, edge_index))
        x = self.feat_dropout(x)

        x = F.relu(self.gat2(x, edge_index))
        x = self.feat_dropout(x)

        x = F.relu(self.gat3(x, edge_index))
        x = self.feat_dropout(x)

        x = self.pool(x, batch)
        return self.lin(x)



class GIN(nn.Module):
    def __init__(self, in_dim=7, hidden=256, out_dim=2, dropout=0):
        super().__init__()

        def make_mlp(in_dim, out_dim):
            return nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.ReLU(),
                nn.Linear(out_dim, out_dim),
            )

        self.conv1 = GINConv(make_mlp(in_dim, hidden))
        self.conv2 = GINConv(make_mlp(hidden, hidden))
        self.conv3 = GINConv(make_mlp(hidden, hidden))
        self.conv4 = GINConv(make_mlp(hidden, hidden))
        self.conv5 = GINConv(make_mlp(hidden, hidden))

        # 🔥 BatchNorm (CRITICAL)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.bn2 = nn.BatchNorm1d(hidden)
        self.bn3 = nn.BatchNorm1d(hidden)
        self.bn4 = nn.BatchNorm1d(hidden)
        self.bn5 = nn.BatchNorm1d(hidden)

        self.dropout = nn.Dropout(dropout)
        self.lin = nn.Linear(hidden, out_dim)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv4(x, edge_index)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv5(x, edge_index)
        x = self.bn5(x)
        x = F.relu(x)

        x = global_mean_pool(x, batch)
        return self.lin(x)


class GCN_SY(nn.Module):
    def __init__(self, in_dim=7, hidden=100, out_dim=2, dropout=0.3):
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
        x = global_add_pool(x, batch)
        return self.lin(x)

class SAGE_SY(nn.Module):
    def __init__(self, in_dim=7, hidden=128, out_dim=2, dropout=0):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden)
        self.conv2 = SAGEConv(hidden, hidden)
        self.conv3 = SAGEConv(hidden, hidden)
        # self.conv4 = SAGEConv(hidden, hidden)
        self.dropout = nn.Dropout(dropout)
        self.lin = nn.Linear(hidden, out_dim)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv3(x, edge_index))
        x = self.dropout(x)
        # x = F.relu(self.conv4(x, edge_index))
        # x = self.dropout(x)
        x = global_add_pool(x, batch)
        return self.lin(x)


class GAT_SY(nn.Module):
    def __init__(self, in_dim=7, hidden=128, heads1=2, heads2=2, out_dim=2, attn_dropout=0, feat_dropout=0):
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
            concat=True,
            dropout=attn_dropout,
        )
        self.gat3 = GATConv(
            in_channels=hidden * heads2,
            out_channels=hidden,
            heads=heads2,
            concat=False,
            dropout=attn_dropout,
        )

        # self.bn1 = nn.BatchNorm1d(hidden * heads1)
        # self.bn2 = nn.BatchNorm1d(hidden)
        # self.bn3 = nn.BatchNorm1d(hidden)

        self.feat_dropout = nn.Dropout(feat_dropout)
        self.pool = global_add_pool
        self.lin = nn.Linear(hidden, out_dim)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.gat1(x, edge_index))
        x = self.feat_dropout(x)

        x = F.relu(self.gat2(x, edge_index))
        x = self.feat_dropout(x)

        x = F.relu(self.gat3(x, edge_index))
        x = self.feat_dropout(x)

        x = self.pool(x, batch)
        return self.lin(x)


class GATv2_SY(nn.Module):
    def __init__(self, in_dim=7, hidden=100, heads1=4, heads2=4, out_dim=2, attn_dropout=0.0, feat_dropout=0.0):
        super().__init__()
        self.gat1 = GATv2Conv(
            in_channels=in_dim,
            out_channels=hidden,
            heads=heads1,
            concat=True,
            dropout=attn_dropout,
        )
        self.gat2 = GATv2Conv(
            in_channels=hidden * heads1,
            out_channels=hidden,
            heads=heads2,
            concat=True,
            dropout=attn_dropout,
        )
        self.gat3 = GATv2Conv(
            in_channels=hidden * heads2,
            out_channels=hidden,
            heads=heads2,
            concat=False,
            dropout=attn_dropout,
        )

        # self.bn1 = nn.BatchNorm1d(hidden * heads1)
        # self.bn2 = nn.BatchNorm1d(hidden)
        # self.bn3 = nn.BatchNorm1d(hidden)

        self.feat_dropout = nn.Dropout(feat_dropout)
        self.pool = global_mean_pool
        self.lin = nn.Linear(hidden, out_dim)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.gat1(x, edge_index))
        x = self.feat_dropout(x)

        x = F.relu(self.gat2(x, edge_index))
        x = self.feat_dropout(x)

        x = F.relu(self.gat3(x, edge_index))
        x = self.feat_dropout(x)

        x = self.pool(x, batch)
        return self.lin(x)


class GIN_SY(nn.Module):
    def __init__(self, in_dim=7, hidden=128, out_dim=2, dropout=0):
        super().__init__()

        nn1 = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
        )
        nn2 = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        nn3 = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        # nn4 = nn.Sequential(
        #     nn.Linear(hidden, hidden),
        #     nn.ReLU(),
        # )
        # nn5 = nn.Sequential(
        #     nn.Linear(hidden, hidden),
        #     nn.ReLU(),
        # )

        self.conv1 = GINConv(nn1)
        self.conv2 = GINConv(nn2)
        self.conv3 = GINConv(nn3)
        # self.conv4 = GINConv(nn4)
        # self.conv5 = GINConv(nn5)

        self.dropout = nn.Dropout(dropout)
        self.lin = nn.Linear(hidden, out_dim)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv3(x, edge_index))
        x = self.dropout(x)
        # x = F.relu(self.conv4(x, edge_index))
        # x = self.dropout(x)
        # x = F.relu(self.conv5(x, edge_index))
        x = global_mean_pool(x, batch)
        return self.lin(x)