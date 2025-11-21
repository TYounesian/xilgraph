import sys

import torch
from torch import nn
from torch_geometric.data import Data
import networkx as nx
from torch_geometric.utils import to_networkx
from networkx.algorithms.isomorphism import GraphMatcher, categorical_node_match
import matplotlib.pyplot as plt
import matplotlib
import random
import numpy as np
from torch_geometric.explain import Explainer
# from graphxai.explainers import GradExplainer, GradCAM
from torch_geometric.explain.algorithm import CaptumExplainer
from torch_geometric.data import Batch
from captum.attr import IntegratedGradients, Saliency
from torch_geometric.nn import to_captum_model, to_captum_input
import numpy as np
from sklearn.metrics import roc_auc_score


def sample_colors(n: int, probs: torch.Tensor) -> torch.Tensor:
    """
    Sample node colors for 4 equally likely colors:
    0='b', 1='r', 2='g', 3='y'.
    Returns a LongTensor (n,) of integer color labels.
    """
    idx = torch.multinomial(probs, num_samples=n, replacement=True)
    return idx.long()


def add_colored_node(edge_index: torch.Tensor,
                     colors: list,
                     color_id: int) -> tuple[torch.Tensor, torch.Tensor, int]:
    """
    Add a single node with given color to the graph and connect it
    with one undirected edge to a random existing node (if any).
    Returns: (new_edge_index, new_colors, new_node_id)
    """
    device = edge_index.device
    n = len(colors)
    new_id = n

    # append color
    new_colors = torch.cat([colors, torch.tensor([color_id], device=device, dtype=torch.long)], dim=0)

    # connect to a random existing node if n>0
    if n > 0:
        attach_to = int(torch.randint(0, n, (1,), device=device))
        attach_edge = torch.tensor([[attach_to, new_id], [new_id, attach_to]],
                                   device=device, dtype=torch.long)
        new_edge_index = torch.cat([edge_index, attach_edge], dim=1)
    else:
        new_edge_index = edge_index

    return new_edge_index, new_colors, new_id


def add_motif_eval(trees: list, edge_index: torch.Tensor, colors: torch.tensor, CID: dict):
    label = int(torch.rand(()) >= 0.5)
    motif_graph = trees[0] if label == 0 else trees[1]

    n = colors.size(0)

    motif_edges = torch.tensor(list(motif_graph.edges)).t().contiguous() + n
    motif_edges = torch.cat([motif_edges, motif_edges.flip(0)], dim=1)

    mc = []
    for u in list(motif_graph.nodes()):
        c = motif_graph.nodes[u].get("color", 0)
        mc.append(CID[c] if isinstance(c, str) else int(c))
    motif_colors = torch.tensor(mc, dtype=torch.long)

    anchor_in_motif = int(torch.randint(0, motif_graph.number_of_nodes(), (1,)))
    attach_target = int(torch.randint(0, n, (1,)))
    attach_edge = torch.tensor([[attach_target], [n + anchor_in_motif]], dtype=torch.long)
    attach_edge = torch.cat([attach_edge, attach_edge.flip(0)], dim=1)

    new_edge_index = torch.cat([edge_index, motif_edges, attach_edge], dim=1)
    new_colors = torch.cat([colors, motif_colors], dim=0)

    motif_node_ids = torch.arange(n, n + len(list(motif_graph.nodes())), dtype=torch.long)
    motif_edge_ids = torch.arange(edge_index.size(1), edge_index.size(1) + motif_edges.size(1), dtype=torch.long)

    return new_edge_index, new_colors, label, motif_node_ids, motif_edge_ids


def add_motif_train_new_color(trees: list, edge_index: torch.Tensor, colors: torch.tensor, CID: dict):
    label = int(torch.rand(()) >= 0.5)
    motif_graph = trees[0] if label == 0 else trees[1]
    target_color = CID["cyan"] if label == 0 else CID["purple"]

    edge_index, colors, new_c_id = add_colored_node(edge_index, colors, target_color)
    n = colors.size(0)

    motif_edges = torch.tensor(list(motif_graph.edges)).t().contiguous() + n
    motif_edges = torch.cat([motif_edges, motif_edges.flip(0)], dim=1)

    mc = []
    for u in list(motif_graph.nodes()):
        c = motif_graph.nodes[u].get("color", 0)
        mc.append(CID[c] if isinstance(c, str) else int(c))
    motif_colors = torch.tensor(mc, dtype=torch.long)

    anchor_in_motif = int(torch.randint(0, motif_graph.number_of_nodes(), (1,)))
    attach_target = (colors == target_color).nonzero(as_tuple=True)[0]
    attach_target = int(attach_target[torch.randint(0, attach_target.numel(), (1,))])
    attach_edge = torch.tensor([[attach_target], [n + anchor_in_motif]], dtype=torch.long)
    attach_edge = torch.cat([attach_edge, attach_edge.flip(0)], dim=1)

    new_edge_index = torch.cat([edge_index, motif_edges, attach_edge], dim=1)
    new_colors = torch.cat([colors, motif_colors], dim=0)

    motif_node_ids = torch.arange(n, n + len(list(motif_graph.nodes())), dtype=torch.long)
    motif_edge_ids = torch.arange(edge_index.size(1), edge_index.size(1) + motif_edges.size(1), dtype=torch.long)

    return new_edge_index, new_colors, label, motif_node_ids, motif_edge_ids, attach_target


def add_motif_train(trees: list, edge_index: torch.Tensor, colors: torch.tensor, target_colors: list, CID: dict):
    label = int(torch.rand(()) >= 0.5)
    motif_graph = trees[0] if label == 0 else trees[1]
    target_color = int(target_colors[0] if label == 0 else target_colors[1])

    edge_index, colors, _ = add_colored_node(edge_index, colors, target_color)
    n = colors.size(0)

    motif_edges = torch.tensor(list(motif_graph.edges)).t().contiguous() + n
    motif_edges = torch.cat([motif_edges, motif_edges.flip(0)], dim=1)

    mc = []
    for u in list(motif_graph.nodes()):
        c = motif_graph.nodes[u].get("color", 0)
        mc.append(CID[c] if isinstance(c, str) else int(c))
    motif_colors = torch.tensor(mc, dtype=torch.long)

    anchor_in_motif = int(torch.randint(0, motif_graph.number_of_nodes(), (1,)))
    attach_target = (colors == target_color).nonzero(as_tuple=True)[0]
    attach_target = int(attach_target[torch.randint(0, attach_target.numel(), (1,))])
    attach_edge = torch.tensor([[attach_target], [n + anchor_in_motif]], dtype=torch.long)
    attach_edge = torch.cat([attach_edge, attach_edge.flip(0)], dim=1)

    new_edge_index = torch.cat([edge_index, motif_edges, attach_edge], dim=1)
    new_colors = torch.cat([colors, motif_colors], dim=0)

    motif_node_ids = torch.arange(n, n + len(list(motif_graph.nodes())), dtype=torch.long)
    motif_edge_ids = torch.arange(edge_index.size(1), edge_index.size(1) + motif_edges.size(1), dtype=torch.long)

    return new_edge_index, new_colors, label, motif_node_ids, motif_edge_ids



def make_graph(trees, G, CID, target_colors, split: str):
    # visualize_graph(edge_index, colors)
    edge_index = torch.tensor(list(G.edges)).t().contiguous()
    colors = torch.tensor([G.nodes[n]["color"] for n in G.nodes], dtype=torch.long)

    if split == "train":
        attach_id = None
        edge_index, colors, y, motif_node_ids, motif_edge_ids, attach_id = add_motif_train_new_color(trees, edge_index, colors, CID)
        # add_motif_eval(trees,
        #                                                                        edge_index,
        #                                                                        colors,
        #                                                                        dict(list(CID.items())[:-2]))

# add_motif_train(trees, edge_index, colors, target_colors, CID)
#
#

    else:
        attach_id = None
        edge_index, colors, y, motif_node_ids, motif_edge_ids = add_motif_eval(trees,
                                                                               edge_index,
                                                                               colors,
                                                                               dict(list(CID.items())[:-2]))

    x = torch.nn.functional.one_hot(colors, num_classes=max(CID.values()) + 1).float()
    data = Data(x=x, edge_index=edge_index)
    data.y = torch.tensor(y, dtype=torch.long)
    data.y_color = colors
    data.split = split
    data.motif_node_ids = motif_node_ids.long().contiguous()
    data.motif_edge_ids = motif_edge_ids.long().contiguous()
    if attach_id is not None:
        data.attach_id = attach_id
    mask = torch.zeros(len(x), dtype=torch.float)
    if hasattr(data, "motif_node_ids"):
        mask[data.motif_node_ids] = 1.0
    data.motif_node_mask = mask

    return data


def visualize_graph(edge_index, colors, title="Graph"):
    """
    Visualize a PyG Data graph with colored nodes.
    `data.y_color` or `data.x` (one-hot) used for node colors.
    """
    x = torch.nn.functional.one_hot(colors, num_classes=4).float()
    data = Data(x=x, edge_index=edge_index)
    # Convert to networkx (undirected)
    G = to_networkx(data, to_undirected=True)

    # Get colors
    if hasattr(data, "y_color"):
        node_colors = data.y_color.cpu().numpy()
    else:
        node_colors = data.x.argmax(dim=1).cpu().numpy()  # infer from one-hot

    # Map numeric color IDs to matplotlib colors
    cmap = {0: "blue", 1: "red", 2: "green", 3: "gold"}
    node_color_list = [cmap.get(int(c), "gray") for c in node_colors]

    # Layout
    pos = nx.spring_layout(G, seed=42)  # deterministic layout

    # Draw
    plt.figure(figsize=(6, 6))
    nx.draw_networkx_nodes(G, pos,
                           node_color=node_color_list,
                           node_size=300,
                           alpha=0.9)
    nx.draw_networkx_edges(G, pos, alpha=0.5, width=1.0)
    nx.draw_networkx_labels(G, pos,
                            labels={i: i for i in range(G.number_of_nodes())},
                            font_size=8)
    plt.title(f"{title}\n(motif: {getattr(data, 'motif', 'N/A')})")
    plt.axis("off")
    plt.show()


def generate_trees(n_tree, tree_colors):
    trees_match = True
    seed = 168013
    while trees_match:
        trees = list()
        T1 = nx.random_tree(n_tree, seed=42)
        for node in T1.nodes:
            random.seed(seed)
            T1.nodes[node]["color"] = random.choice(tree_colors)
            seed += 2

        trees.append(T1)

        T2 = nx.random_tree(n_tree, seed=168012)
        for node in T2.nodes:
            random.seed(seed)
            T2.nodes[node]["color"] = random.choice(tree_colors)
            seed += 2

        # check if T1 and T2 are different
        matcher = nx.algorithms.isomorphism.GraphMatcher(T1, T2,
                                                         node_match=lambda n1, n2: n1['color'] == n2['color'])
        trees_match = len(list(matcher.subgraph_isomorphisms_iter())) > 0

        trees.append(T2)

    return trees


def generate_and_check(trees, n_nodes, p_edge, colors):
    H_T_match = True
    T0 = trees[0]
    T1 = trees[1]

    # generate the ER graph
    while H_T_match:
        G = nx.erdos_renyi_graph(n_nodes, p_edge)
        for node in G.nodes:
            G.nodes[node]["color"] = random.choice(colors)

        largest_cc_nodes = max(nx.connected_components(G), key=len)
        H = G.subgraph(largest_cc_nodes).copy()
        H = nx.convert_node_labels_to_integers(H, first_label=0, ordering="sorted")
        colors = torch.tensor([H.nodes[i]["color"] for i in range(H.number_of_nodes())], dtype=torch.long)

        matcher0 = nx.algorithms.isomorphism.GraphMatcher(H, T0,
                                                         node_match=lambda n1, n2: n1['color'] == n2['color'])
        matches0 = list(matcher0.subgraph_isomorphisms_iter())

        matcher1 = nx.algorithms.isomorphism.GraphMatcher(H, T1,
                                                          node_match=lambda n1, n2: n1['color'] == n2['color'])
        matches1 = list(matcher1.subgraph_isomorphisms_iter())

        if len(matches0)+len(matches1) == 0:
            H_T_match = False

    n = len(H.nodes)
    ei = torch.tensor(list(H.edges)).t().contiguous().max()
    if ei >= n:
        print("higher")
        sys.exit()
    # print(f"Found {len(matches0)} and {len(matches1)} occurrences of the pattern for class 1 and 2.")
    return H


def topk_hit(node_imp: torch.Tensor, motif_nodes: torch.Tensor, k: int = None):
    """
    node_imp: (N,) importance scores
    motif_nodes: (M,) long tensor of node indices
    k: number of nodes to select (defaults to M)
    Returns: precision@k, recall@k, f1@k
    """
    N = node_imp.numel()
    motif_nodes = motif_nodes.long().unique()
    M = motif_nodes.numel()
    k = M if (k is None) else min(k, N)

    topk_idx = torch.topk(node_imp, k=k).indices
    hits = torch.isin(topk_idx, motif_nodes).sum().item()
    precision = hits / max(k, 1)
    recall = hits / max(M, 1)
    f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def iou_at_threshold(node_imp: torch.Tensor, motif_nodes: torch.Tensor, thresh: float):
    """
    Binarize importance by threshold (absolute or percentile you choose beforehand).
    Returns IoU (Jaccard) between predicted set and motif set.
    """
    motif_mask = torch.zeros_like(node_imp, dtype=torch.bool)
    motif_mask[motif_nodes.long().unique()] = True

    pred_mask = node_imp >= thresh
    inter = (pred_mask & motif_mask).sum().item()
    union = (pred_mask | motif_mask).sum().item()
    return inter / max(union, 1)


def average_precision(node_imp: torch.Tensor, motif_nodes: torch.Tensor):
    """
    Compute Average Precision (AP) without sklearn.
    """
    N = node_imp.numel()
    y = torch.zeros(N, dtype=torch.float, device=node_imp.device)
    y[motif_nodes.long().unique()] = 1.0

    # sort by score desc
    scores, order = torch.sort(node_imp, descending=True)
    y_sorted = y[order]

    # precision at each positive
    cum_tp = torch.cumsum(y_sorted, dim=0)
    denom = torch.arange(1, N+1, device=node_imp.device, dtype=torch.float)
    precision = cum_tp / denom
    ap = (precision * y_sorted).sum() / max(y.sum(), torch.tensor(1.0, device=y.device))
    return ap.item()


def captum_explain_graphs(model, graphs, num_samples=5, method="IntegratedGradients"):
        # explainer = Explainer(
        #     model=model,
        #     algorithm=CaptumExplainer(method),  # "IntegratedGradients" | "Saliency" | "DeepLift" | ...
        #     explanation_type='model',
        #     node_mask_type='attributes',  # node-feature attribution
        #     edge_mask_type='object',  # edge mask attribution
        #     model_config=dict(
        #         mode='multiclass_classification',
        #         task_level='graph',
        #         return_type='raw',  # model returns logits
        #     ),
        # )
        total_hit_n = 0.
        total_hit_e = 0.
        graphs_iter = graphs if isinstance(graphs, (list, tuple)) else [graphs]
        for i, g in enumerate(graphs_iter[:min(num_samples, len(graphs_iter))]):
            b = Batch.from_data_list([g])

            mask_type = "node"
            captum_model = to_captum_model(model, mask_type)
            inputs, additional_forward_args = to_captum_input(b.x,
                                                              b.edge_index, mask_type)

            additional_forward_args = (*additional_forward_args, b.batch)

            ig = Saliency(captum_model)
            ig_attr = ig.attribute(inputs=inputs,
                                   target=int(b.y),
                                   additional_forward_args=additional_forward_args) #,
                                   #internal_batch_size=1)
            #
            # exp = explainer(
            #     x=b.x,
            #     edge_index=b.edge_index,
            #     batch=b.batch,
            #     target=target,
            # )
    #
            node_imp = (ig_attr[0].squeeze().pow(2).sum(dim=1) + 1e-9).sqrt() #ig_attr[0].squeeze().abs().sum(dim=1) #exp.node_mask.abs().sum(dim=1)  # aggregate feature importance → [N]
            m, M = node_imp.min().detach(), node_imp.max().detach()
            node_imp = (node_imp - m) / (M - m + node_imp)
            # node_imp = (node_imp - node_imp.min()) / (node_imp.max() - node_imp.min() + 1e-12)
            # edge_imp = exp.edge_mask.detach().cpu()
            topk_nodes = torch.topk(node_imp, k=max(1, int(0.2 * node_imp.numel()))).indices.tolist()
            # topk_edges = torch.topk(edge_imp, k=min(10, edge_imp.numel())).indices.tolist()
            # print(f"[Captum][Graph {i}] target={target} | top nodes: {topk_nodes} | top edge idx: {topk_edges}")

            # check overlap:

            if hasattr(g, "motif_node_ids"):
                motif_n = torch.as_tensor(g.motif_node_ids)
                hit_n = torch.isin(motif_n, torch.as_tensor(topk_nodes)).sum().item()/len(motif_n)
                total_hit_n += hit_n
                motif_e = torch.as_tensor(g.motif_edge_ids)
                hit_e = 0 #torch.isin(torch.as_tensor(topk_edges), motif_e).float().mean().item()
                total_hit_e += hit_e
                # plot_node_importance(g, motif_n, node_imp, title="Captum Node Importance")
                # print(f"motif node hit@top20% = {hit_n:.3f}, motif edge hit@top20% = {hit_e:.3f}")
            # if hasattr(g, "attach_id"):
            #     attach_n = torch.as_tensor(g.attach_id)
            #     hit_n = sum(torch.isin(torch.as_tensor(topk_nodes), attach_n)).float()
            #     total_hit_n += hit_n
            #     print(f"Label: {g.y}, attach node hit@top20% = {hit_n:.3f}")

        return node_imp, total_hit_n/num_samples, total_hit_e/num_samples


def grad_explainer(model, graphs, trees):
    # grd_exp = GradExplainer(model=model, criterion=nn.CrossEntropyLoss())
    grd_exp = GradCAM(model=model, criterion=nn.CrossEntropyLoss())

    total_f1 = 0.
    total_r = 0.
    num_samples = len(graphs['train'])
    for candidate_g in graphs['train']:
        batch = torch.zeros(candidate_g.num_nodes, dtype=torch.long, device=candidate_g.x.device)
        exp = grd_exp.get_explanation_graph(candidate_g.x, candidate_g.edge_index, candidate_g.y, batch)

        label = int(candidate_g.y.item())

        # select the correct motif nodes based on the label
        if hasattr(candidate_g, "attach_id"):
            motif_nodes = torch.as_tensor(candidate_g.attach_id)
        else:
            motif_graph = trees[label]  # tree[0] if label==0 else tree[1]
            motif_nodes = torch.arange(
                candidate_g.num_nodes - motif_graph.number_of_nodes(),
                candidate_g.num_nodes
            )

        # evaluate explanation performance
        node_imp = exp.node_imp.view(-1)
        p, r, f1 = topk_hit(node_imp, motif_nodes, int(0.2 * candidate_g.num_nodes))
        ap = average_precision(node_imp, motif_nodes)
        # print(f"Label={label} | P={p:.3f} R={r:.3f} F1={f1:.3f} | AP={ap:.3f}")
        total_f1 += f1
        total_r += r
        # plot_node_importance(candidate_g, node_imp, title="Grad Node Importance")

    return total_f1/num_samples, total_r/num_samples


def plot_node_importance(graph, motif_nodes, node_imp, title="Node importance"):
    """
    Visualize a graph with nodes colored by importance scores.

    Args:
        graph: PyG Data object or NetworkX Graph
        node_imp (torch.Tensor or list): importance per node (len = num_nodes)
        title (str): plot title
    """
    # Convert PyG → NetworkX if needed
    if not isinstance(graph, nx.Graph):
        G = to_networkx(graph, to_undirected=True)
    else:
        G = graph

    # convert importance tensor to numpy
    node_imp = torch.as_tensor(node_imp, dtype=torch.float).detach().cpu()
    node_imp = (node_imp - node_imp.min()) / (node_imp.max() - node_imp.min() + 1e-9)  # normalize 0–1

    # assign as node attributes for plotting
    for i, score in enumerate(node_imp.tolist()):
        G.nodes[i]["importance"] = score

    motif_nodes = torch.as_tensor(motif_nodes, dtype=torch.long).detach().cpu().unique()

    # get colors and layout
    colors = [G.nodes[i]["importance"] for i in G.nodes()]
    pos = nx.spring_layout(G, seed=42)

    plt.figure(figsize=(6, 5))
    norm = matplotlib.colors.Normalize(vmin=min(colors), vmax=max(colors))
    mapper = plt.cm.ScalarMappable(norm=norm, cmap=plt.cm.viridis)
    face_colors = [mapper.to_rgba(c) for c in colors]

    nx.draw(G, pos, node_color=face_colors, with_labels=False,
            node_size=300, edge_color="#888")

    motif_list = motif_nodes.tolist()
    if len(motif_list):
        nx.draw_networkx_nodes(
            G, pos, nodelist=motif_list,
            node_color=[face_colors[i] for i in motif_list],
            node_size=420, linewidths=2.5, edgecolors="crimson"
        )

    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis)
    sm.set_array(colors)
    plt.colorbar(sm, label="Importance")
    plt.title(title)
    plt.axis("off")
    plt.show()


def run_epoch(model, loader, opt, criterion, train: bool, device="cpu"):
    if train:
        model.train()
    else:
        model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    for batch in loader:
        batch = batch.to(device)
        if train:
            opt.zero_grad()
        out = model(batch.x, batch.edge_index, batch.batch)
        loss = criterion(out, batch.y.view(-1))
        if train:
            loss.backward()
            opt.step()
        preds = out.argmax(dim=-1)
        correct += (preds == batch.y.view(-1)).sum().item()
        total += batch.y.size(0)
        loss_sum += loss.item() * batch.y.size(0)
    acc = correct / max(total, 1)
    avg_loss = loss_sum / max(total, 1)
    return avg_loss, acc


def plot_g_tree(g, trees, CID):
    cmap = {v: k for k, v in CID.items()}

    # to plot the graph and the tree
    def _node_colors(G):
        # each node has an integer color ID
        node_color_list = [cmap.get(int(G.nodes[n].get("color", 0)), "gray") for n in G.nodes()]
        return node_color_list

    G = nx.Graph()
    num_nodes = g.num_nodes
    G.add_nodes_from(range(num_nodes))
    u, v = g.edge_index.cpu().numpy()
    edges = list(zip(u, v))
    G.add_edges_from(edges)
    for i, c in enumerate(g.y_color.cpu().tolist()):
        G.nodes[i]["color"] = int(c)

    posH = nx.spring_layout(G, seed=42)
    posT = nx.spring_layout(trees[g.y], seed=42)
    cH = _node_colors(G)
    cT = _node_colors(trees[g.y])

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    nx.draw(G, posH, node_color=cH, with_labels=False, node_size=250, edge_color="#888")
    plt.title('H')
    plt.axis("off")

    plt.subplot(1, 2, 2)
    nx.draw(trees[g.y], posT, node_color=cT, with_labels=False, node_size=250, edge_color="#888")
    plt.title('tree')
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def saliency_to_probs_single(node_imp: torch.Tensor, tau: float = 0.25):
    return torch.softmax(node_imp / tau, dim=0)


def soft_target_from_mask_single(mask: torch.Tensor, eps: float = 1e-9):
    mask = mask.bool()
    q = torch.zeros_like(mask, dtype=torch.float)
    q[mask] = 1.0 / mask.sum()
    return q.clamp_min(eps)


def saliency_grad_diff(model, batch):
    # model.eval()
    x = batch.x.clone().requires_grad_(True)

    logits = model(x, batch.edge_index, batch.batch)
    B, C = logits.shape
    target = batch.y.to(logits.device).long()
    idx = torch.arange(B, device=logits.device)

    scalar = logits[idx, target].sum()
    grads = torch.autograd.grad(
        scalar, x,
        create_graph=True,
        retain_graph=True
    )[0]
    # model.train()

    node_imp = (grads.pow(2).sum(dim=1) + 1e-9).sqrt()# [N], raw real-valued importance

    # topk_nodes = torch.topk(node_imp2, k=max(1, int(0.2 * node_imp2.numel()))).indices.tolist()

    hits = []
    aucs = []
    node_imp2 = node_imp.clone()
    for g_id in batch.batch.unique():
        m = (batch.batch == g_id)  # nodes of this graph
        motif_mask_g = batch.motif_node_mask[m].bool()
        motif_idx_g = motif_mask_g.nonzero(as_tuple=True)[0]

        mi, ma = node_imp2[m].min().detach(), node_imp2[m].max().detach()
        node_imp2[m] = (node_imp2[m] - mi) / (ma - mi + 1e-8)
        topk_local = torch.topk(node_imp2[m], k=max(1, int(0.2 * node_imp2[m].numel()))).indices

        hit_n = torch.isin(motif_idx_g, topk_local).float().mean().item()
        hits.append(hit_n)

        auc = roc_auc_score(motif_mask_g.cpu().numpy().astype(np.int32), node_imp[m].cpu().detach().numpy().astype(np.float32))
        aucs.append(auc)

    # if hasattr(g, "motif_node_ids"):
    #     motif_n = torch.as_tensor(g.motif_node_ids)
    #     hit_n = torch.isin(motif_n, torch.as_tensor(topk_nodes)).sum().item() / len(motif_n)

    saliency = grads.abs()
    return node_imp2, saliency, sum(hits)/len(hits), float(np.mean(aucs))
