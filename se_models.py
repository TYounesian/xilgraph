import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GCNConv, GATConv, GATv2Conv, GINConv, global_mean_pool, global_add_pool


def get_architecture(in_dim, out_dim, architecture):
    if architecture.upper() == "GIN":
        layer = GINConv(
            nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.ReLU(),
            )
        )
    elif architecture.upper()== "SAGE":
        layer = SAGEConv(in_dim, out_dim)
    elif architecture.upper()== "GCN":
        layer = GCNConv(in_dim, out_dim)
    elif architecture.upper()== "GAT":
        layer = GATv2Conv(
            in_channels=in_dim,
            out_channels=out_dim,
            heads=2,
            concat=False,
            dropout=0.0,
        )
    return layer

class SEGAT(nn.Module):
    def __init__(self, disable_expl, in_dim=7, hidden=100, out_dim=2, dropout=0):
        super().__init__()

        self.disable_expl = disable_expl
        self.explainer = SelfExplainer(in_dim, hidden, num_layers=3, architecture="GAT")

        self.conv1 = get_architecture(in_dim, hidden, "GAT")
        self.conv2 = get_architecture(hidden, hidden, "GAT")
        self.conv3 = get_architecture(hidden, hidden, "GAT")
        self.dropout = nn.Dropout(dropout)
        self.lin = nn.Linear(hidden, out_dim)

    def forward(self, x, edge_index, batch):
        if self.disable_expl:
            attn_logit = torch.zeros((x.shape[0], 1), device=x.device)
            attn_prob  = torch.ones((x.shape[0], 1), device=x.device)
        else:
            attn_logit = self.explainer(x, edge_index, batch)
            attn_prob = torch.sigmoid(attn_logit) # shape: Bx1
            attn_prob = attn_prob.detach() # important to avoid label-encoding explanaitons

        x = F.relu(self.conv1(x * attn_prob, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv2(x * attn_prob, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv3(x * attn_prob, edge_index))
        x = self.dropout(x)
        x = global_mean_pool(x * attn_prob, batch)
        return attn_logit.squeeze(1), self.lin(x)
    
class SEGCN(nn.Module):
    def __init__(self, disable_expl, in_dim=7, hidden=100, out_dim=2, dropout=0):
        super().__init__()

        self.disable_expl = disable_expl
        self.explainer = SelfExplainer(in_dim, hidden, num_layers=3, architecture="GCN")

        self.conv1 = get_architecture(in_dim, hidden, "GCN")
        self.conv2 = get_architecture(hidden, hidden, "GCN")
        self.conv3 = get_architecture(hidden, hidden, "GCN")
        self.dropout = nn.Dropout(dropout)
        self.lin = nn.Linear(hidden, out_dim)

    def forward(self, x, edge_index, batch):
        if self.disable_expl:
            attn_logit = torch.zeros((x.shape[0], 1), device=x.device)
            attn_prob  = torch.ones((x.shape[0], 1), device=x.device)
        else:
            attn_logit = self.explainer(x, edge_index, batch)
            attn_prob = torch.sigmoid(attn_logit) # shape: Bx1
            attn_prob = attn_prob.detach() # important to avoid label-encoding explanaitons

        x = F.relu(self.conv1(x * attn_prob, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv2(x * attn_prob, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv3(x * attn_prob, edge_index))
        x = self.dropout(x)
        x = global_mean_pool(x * attn_prob, batch)
        return attn_logit.squeeze(1), self.lin(x)
    

class SESAGE(nn.Module):
    def __init__(self, disable_expl, in_dim=7, hidden=100, out_dim=2, dropout=0):
        super().__init__()

        self.disable_expl = disable_expl
        self.explainer = SelfExplainer(in_dim, hidden, num_layers=3, architecture="SAGE")

        self.conv1 = get_architecture(in_dim, hidden, "SAGE")
        self.conv2 = get_architecture(hidden, hidden, "SAGE")
        self.conv3 = get_architecture(hidden, hidden, "SAGE")
        self.dropout = nn.Dropout(dropout)
        self.lin = nn.Linear(hidden, out_dim)

    def forward(self, x, edge_index, batch):
        if self.disable_expl:
            attn_logit = torch.zeros((x.shape[0], 1), device=x.device)
            attn_prob  = torch.ones((x.shape[0], 1), device=x.device)
        else:
            attn_logit = self.explainer(x, edge_index, batch)
            attn_prob = torch.sigmoid(attn_logit) # shape: Bx1
            attn_prob = attn_prob.detach() # important to avoid label-encoding explanaitons

        x = F.relu(self.conv1(x * attn_prob, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv2(x * attn_prob, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv3(x * attn_prob, edge_index))
        x = self.dropout(x)
        x = global_mean_pool(x * attn_prob, batch)
        return attn_logit.squeeze(1), self.lin(x)

class SEGIN(nn.Module):
    def __init__(self, disable_expl, in_dim=7, hidden=10, out_dim=2, dropout=0):
        super().__init__()

        self.disable_expl = disable_expl
        self.explainer = SelfExplainer(in_dim, hidden, num_layers=3, architecture="GIN")

        self.conv1 = get_architecture(in_dim, hidden, "GIN") # GINConv(nn1)
        self.conv2 = get_architecture(hidden, hidden, "GIN") # GINConv(nn2)
        self.conv3 = get_architecture(hidden, hidden, "GIN") # GINConv(nn3)
        self.dropout = nn.Dropout(dropout)
        self.lin = nn.Linear(hidden, out_dim)

    def forward(self, x, edge_index, batch):
        if self.disable_expl:
            attn_logit = torch.zeros((x.shape[0], 1), device=x.device)
            attn_prob  = torch.ones((x.shape[0], 1), device=x.device)
        else:
            attn_logit = self.explainer(x, edge_index, batch)
            attn_prob = torch.sigmoid(attn_logit) # shape: Bx1
            attn_prob = attn_prob.detach() # important to avoid label-encoding explanaitons

        x = F.relu(self.conv1(x * attn_prob, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv2(x * attn_prob, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv3(x * attn_prob, edge_index))
        x = self.dropout(x)
        x = global_mean_pool(x * attn_prob, batch)
        return attn_logit.squeeze(1), self.lin(x)
    
class SelfExplainer(nn.Module):
    def __init__(self, in_dim, hidden, num_layers, architecture, attn_type="node"):
        super().__init__()

        self.encoder = nn.ModuleList()
        self.architecture = architecture
        self.attn_type = attn_type

        for i in range(num_layers):
            self.encoder.append(
                get_architecture(
                    in_dim=in_dim if i == 0 else hidden,
                    out_dim=hidden,
                    architecture=architecture
                )
            )
        
        self.head = nn.Linear(hidden if attn_type == "node" else hidden * 2, 1, bias=False)

    def forward(self, x, edge_index, batch):        
        embed = x
        for layer in self.encoder:
            embed = layer(embed, edge_index)
            embed = F.relu(embed)
        
        if self.attn_type == "node":
            attn_logit = self.head(embed)
        elif self.attn_type == "edge":
            col, row = edge_index
            f1, f2 = embed[col], embed[row]
            f12 = torch.cat([f1, f2], dim=-1) # attention scores are not symmetric!
            attn_logit = self.head(f12)
        
        return attn_logit

# class GAT(nn.Module):
#     def __init__(self, in_dim=7, hidden=100, heads1=4, heads2=4, out_dim=2, attn_dropout=0.0, feat_dropout=0.0):
#         super().__init__()
#         self.gat1 = GATConv(
#             in_channels=in_dim,
#             out_channels=hidden,
#             heads=heads1,
#             concat=True,
#             dropout=attn_dropout,
#         )
#         self.gat2 = GATConv(
#             in_channels=hidden * heads1,
#             out_channels=hidden,
#             heads=heads2,
#             concat=True,
#             dropout=attn_dropout,
#         )
#         self.gat3 = GATConv(
#             in_channels=hidden * heads2,
#             out_channels=hidden,
#             heads=heads2,
#             concat=False,
#             dropout=attn_dropout,
#         )

#         # self.bn1 = nn.BatchNorm1d(hidden * heads1)
#         # self.bn2 = nn.BatchNorm1d(hidden)
#         # self.bn3 = nn.BatchNorm1d(hidden)

#         self.feat_dropout = nn.Dropout(feat_dropout)
#         self.pool = global_mean_pool
#         self.lin = nn.Linear(hidden, out_dim)

#     def forward(self, x, edge_index, batch):
#         x = F.relu(self.gat1(x, edge_index))
#         x = self.feat_dropout(x)

#         x = F.relu(self.gat2(x, edge_index))
#         x = self.feat_dropout(x)

#         x = F.relu(self.gat3(x, edge_index))
#         x = self.feat_dropout(x)

#         x = self.pool(x, batch)
#         return self.lin(x)


# class GATv2(nn.Module):
#     def __init__(self, in_dim=7, hidden=100, heads1=4, heads2=4, out_dim=2, attn_dropout=0.0, feat_dropout=0.0):
#         super().__init__()
#         self.gat1 = GATv2Conv(
#             in_channels=in_dim,
#             out_channels=hidden,
#             heads=heads1,
#             concat=True,
#             dropout=attn_dropout,
#         )
#         self.gat2 = GATv2Conv(
#             in_channels=hidden * heads1,
#             out_channels=hidden,
#             heads=heads2,
#             concat=True,
#             dropout=attn_dropout,
#         )
#         self.gat3 = GATv2Conv(
#             in_channels=hidden * heads2,
#             out_channels=hidden,
#             heads=heads2,
#             concat=False,
#             dropout=attn_dropout,
#         )

#         # self.bn1 = nn.BatchNorm1d(hidden * heads1)
#         # self.bn2 = nn.BatchNorm1d(hidden)
#         # self.bn3 = nn.BatchNorm1d(hidden)

#         self.feat_dropout = nn.Dropout(feat_dropout)
#         self.pool = global_mean_pool
#         self.lin = nn.Linear(hidden, out_dim)

#     def forward(self, x, edge_index, batch):
#         x = F.relu(self.gat1(x, edge_index))
#         x = self.feat_dropout(x)

#         x = F.relu(self.gat2(x, edge_index))
#         x = self.feat_dropout(x)

#         x = F.relu(self.gat3(x, edge_index))
#         x = self.feat_dropout(x)

#         x = self.pool(x, batch)
#         return self.lin(x)