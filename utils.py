import pdb
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
import torch.nn.functional as F
from torch_geometric.utils import softmax



SEED = 42
random.seed(SEED)
np.random.seed(SEED)


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
    k = random.choice(range(1,25))
    new_ids = torch.arange(k) + n

    # append color
    new_colors = torch.cat([colors, color_id*torch.ones_like(new_ids, device=device, dtype=torch.long)], dim=0)

    # connect to a random existing node if n>0
    if n > 0:
        attach_to = torch.randint(0, n, (k,), device=device)
        attach_edge = torch.stack([torch.cat([attach_to, new_ids]), torch.cat([new_ids, attach_to])], dim=0)
        new_edge_index = torch.cat([edge_index, attach_edge], dim=1)
    else:
        new_edge_index = edge_index

    return new_edge_index, new_colors, new_ids


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

    edge_index, colors, conf_id = add_colored_node(edge_index, colors, target_color)
    n = colors.size(0)

    motif_edges = torch.tensor(list(motif_graph.edges)).t().contiguous() + n
    motif_edges = torch.cat([motif_edges, motif_edges.flip(0)], dim=1)

    mc = []
    for u in list(motif_graph.nodes()):
        c = motif_graph.nodes[u].get("color", 0)
        mc.append(CID[c] if isinstance(c, str) else int(c))
    motif_colors = torch.tensor(mc, dtype=torch.long)

    anchor_in_motif = int(torch.randint(0, motif_graph.number_of_nodes(), (1,)))
    # attach_target = (colors == target_color).nonzero(as_tuple=True)[0]
    # attach_target = int(attach_target[torch.randint(0, attach_target.numel(), (1,))])
    attach_target = int(torch.randint(0, n, (1,)))
    attach_edge = torch.tensor([[attach_target], [n + anchor_in_motif]], dtype=torch.long)
    attach_edge = torch.cat([attach_edge, attach_edge.flip(0)], dim=1)

    new_edge_index = torch.cat([edge_index, motif_edges, attach_edge], dim=1)
    new_colors = torch.cat([colors, motif_colors], dim=0)

    motif_node_ids = torch.arange(n, n + len(list(motif_graph.nodes())), dtype=torch.long)
    motif_edge_ids = torch.arange(edge_index.size(1), edge_index.size(1) + motif_edges.size(1), dtype=torch.long)

    return new_edge_index, new_colors, label, motif_node_ids, motif_edge_ids, attach_target, conf_id


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


def make_graph(trees, G, CID, target_colors, split, confounder_flag):
    # visualize_graph(edge_index, colors)
    edge_index = torch.tensor(list(G.edges)).t().contiguous()
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    colors = torch.tensor([G.nodes[n]["color"] for n in G.nodes], dtype=torch.long)

    if split == "train":
        attach_id = None #1000
        conf_id = None #torch.tensor([1000])
        if confounder_flag:
            edge_index, colors, y, motif_node_ids, motif_edge_ids, attach_id, conf_id = add_motif_train_new_color(trees, edge_index, colors, CID)
        else:
            edge_index, colors, y, motif_node_ids, motif_edge_ids = add_motif_eval(trees,
                                                                                   edge_index,
                                                                                   colors,
                                                                                   dict(list(CID.items())[:-2]))
        # add_motif_eval(trees,
        #                                                                        edge_index,
        #                                                                        colors,
        #                                                                        dict(list(CID.items())[:-2]))

# add_motif_train(trees, edge_index, colors, target_colors, CID)
#
#

    else:
        attach_id = None
        conf_id = None
        edge_index, colors, y, motif_node_ids, motif_edge_ids = add_motif_eval(trees,
                                                                               edge_index,
                                                                               colors,
                                                                               dict(list(CID.items())[:-2]))

    x = torch.nn.functional.one_hot(colors, num_classes=max(CID.values()) + 1).float()
    if split == 'train' and confounder_flag == True:
        x[:,-2:] *= 100
    data = Data(x=x, edge_index=edge_index)
    data.y = torch.tensor(y, dtype=torch.long)
    data.y_color = colors
    data.split = split
    data.motif_node_ids = motif_node_ids.long().contiguous()
    data.motif_edge_ids = motif_edge_ids.long().contiguous()
    if split == 'train':
        data.conf_id = None if conf_id is None else conf_id.long().contiguous()
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


def plot_node_importance(graph, motif_nodes, conf_id, node_imp, title="Node importance"):
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
    node_imp_c = torch.as_tensor(node_imp, dtype=torch.float).detach().cpu()
    # node_imp = (node_imp - node_imp.min()) / (node_imp.max() - node_imp.min() + 1e-9)  # normalize 0–1

    # assign as node attributes for plotting
    for i, score in enumerate(node_imp_c.tolist()):
        G.nodes[i]["importance"] = score

    motif_nodes = torch.as_tensor(motif_nodes, dtype=torch.long).detach().cpu().unique()

    # get colors and layout
    colors = [G.nodes[i]["importance"] for i in G.nodes()]
    coords = graph.x[:, -2:].cpu().numpy()
    pos = {i: coords[i] for i in range(coords.shape[0])}

    # pos = nx.spring_layout(G, seed=42)

    plt.figure(figsize=(6, 5))
    norm = matplotlib.colors.Normalize(vmin=min(colors), vmax=max(colors))
    mapper = plt.cm.ScalarMappable(norm=norm, cmap=plt.cm.plasma)
    face_colors = [mapper.to_rgba(c) for c in colors]

    nx.draw(G, pos, node_color=face_colors, with_labels=False,
            node_size=300, edge_color="#888")

    motif_list = motif_nodes.tolist()
    if conf_id is not None:
        conf_list = conf_id.detach().cpu().tolist()
    else:
        conf_list = []

    if len(motif_list):
        nx.draw_networkx_nodes(
            G, pos, nodelist=motif_list,
            node_color=[face_colors[i] for i in motif_list],
            node_size=420, linewidths=2.5, edgecolors="cyan"
        )

    if len(conf_list):
        nx.draw_networkx_nodes(
            G, pos, nodelist=conf_list,
            node_color=[face_colors[i] for i in conf_list],
            node_size=420, linewidths=2.5, edgecolors="green"
        )

    sm = plt.cm.ScalarMappable(cmap=plt.cm.plasma)
    sm.set_array(colors)
    plt.colorbar(sm, ax=plt.gca(), label="Importance")
    plt.title(title)
    plt.axis("off")
    plt.gca().invert_yaxis()
    plt.show()


def run_epoch(model, loader, opt, criterion, epoch, train: bool, device="cpu"):
    if train:
        model.train()
    else:
        model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    cnttt = 0.
    for batch in loader:
        # batch.x = batch.x[:, 3:]
        batch = batch.to(device)
        if train:
            opt.zero_grad()
        out = model(batch.x, batch.edge_index, batch.batch)

        if type(out) is tuple:
            expl_attn_logit, out = out # separate explanation from target predictions

        # plot all instances
        # _, sal_b, _, _ = saliency_grad_diff(model, batch)
        # node_imp_b = sal_b.sum(dim=1)
        # chosen_mask = torch.ones(batch.y.view(-1).size(0))
        # node_mask = chosen_mask[batch.batch].bool()
        # gt_mask = batch.motif_node_mask[node_mask].float()
        # pos_loss = -torch.mean(node_imp_b[gt_mask.bool()])
        # # Negative mask: want low saliency
        # neg_loss = torch.mean(node_imp_b[~gt_mask.bool()])
        # expl_loss = pos_loss+neg_loss
        data_list = batch.to_data_list()

        # Plot each graph using ptr to slice node_imp correctly
        # if train and epoch > 2:
        #     for i, g in enumerate(data_list):
        #         if i < 4 and cnttt < 2:
        #             start, end = int(batch.ptr[i]), int(batch.ptr[i + 1])
        #             node_imp_g = node_imp_b[start:end].detach().cpu()
        #             print("pos avg:", node_imp_g[g.motif_node_mask.bool()].mean().item())
        #             print("conf avg:", node_imp_g[g.conf_id].mean().item())
        #             print("neg avg:", node_imp_g[~g.motif_node_mask.bool()].mean().item())
        #
        #             plot_node_importance(
        #                 g,
        #                 g.motif_node_ids,
        #                 g.conf_id,
        #                 node_imp_g,
        #                 title=f"Node Importance (graph {i})",
        #             )
        # cnttt += 1
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


def plot_g_tree(g, trees, CID, node_imp=None):
    cmap = {v: k for k, v in CID.items()}

    def _node_colors(G):
        return [cmap.get(int(G.nodes[n].get("color", 0)), "gray") for n in G.nodes()]

    G = nx.Graph()
    num_nodes = g.num_nodes
    G.add_nodes_from(range(num_nodes))
    u, v = g.edge_index.cpu().numpy()
    G.add_edges_from(list(zip(u, v)))
    for i, c in enumerate(g.y_color.cpu().tolist()):
        G.nodes[i]["color"] = int(c)

    posH = nx.spring_layout(G, seed=42)
    posT = nx.spring_layout(trees[g.y], seed=42)
    cH = _node_colors(G)
    cT = _node_colors(trees[g.y])

    # ---- TOP-1 importance nodes (in H) ----
    top2 = set()
    if node_imp is not None:
        imp = node_imp.detach().numpy()
        top2 = set(np.argsort(imp)[-1:])  # indices of top 2

    # Degrees in the graph
        print(
            f"avg degree: {sum(dict(G.degree()).values()) / G.number_of_nodes():.2f}, max degree: {max(dict(G.degree()).values())}")
        print(f"Top-1 important node's degree: {G.degree[int(np.argsort(imp)[-1:])]}")
        print(f"rank of highest-degree node (by importance): {len(imp) - np.argsort(imp).tolist().index(max(G.degree, key=lambda x: x[1])[0])}")

    #get the norm_mass
    # edge_index: [2, num_edges]
    src, dst = g.edge_index

    num_nodes = g.num_nodes

    # degree (undirected or directed-in, depending on your graph)
    deg = torch.zeros(num_nodes, device=src.device)
    deg.index_add_(0, src, torch.ones_like(src, dtype=deg.dtype))
    # deg.index_add_(0, dst, torch.ones_like(dst, dtype=deg.dtype))  # remove if directed

    deg1 = torch.tensor([G.degree[n] for n in range(G.number_of_nodes())], dtype=torch.float)

    # normalized mass
    norm_mass = torch.zeros(num_nodes, device=src.device)

    contrib = 1.0 / torch.sqrt(deg[src] * deg[dst])
    norm_mass.index_add_(0, dst, contrib)

    # highest norm_mass node
    top_nm_node = torch.argmax(norm_mass).item()

    # its node_imp rank (1 = highest importance)
    rank = (torch.argsort(node_imp, descending=True) == top_nm_node).nonzero(as_tuple=True)[0].item() + 1

    print(f"lowest norm_mass node_imp rank: {rank}")

    # highest node_imp node
    top_imp_node = torch.argmax(node_imp).item()

    # its norm_mass rank (1 = lowest norm_mass)
    rank_nm = (torch.argsort(norm_mass, descending=True) == top_imp_node).nonzero(as_tuple=True)[0].item() + 1

    print(f"highest node_imp norm_mass rank: {rank_nm}")

    plt.scatter(norm_mass.detach().numpy(), node_imp.detach(). numpy())
    plt.xlabel("norm_mass")
    plt.ylabel("node_imp")
    plt.show()

    # node borders for H
    border_colors_H = ["red" if n in top2 else "black" for n in G.nodes()]
    border_widths_H = [2.5 if n in top2 else 0.5 for n in G.nodes()]

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    nx.draw(
        G, posH,
        node_color=cH,
        edgecolors=border_colors_H,
        linewidths=border_widths_H,
        with_labels=False, node_size=250, edge_color="#888"
    )
    plt.title("H")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    nx.draw(
        trees[g.y], posT,
        node_color=cT,
        with_labels=False, node_size=250, edge_color="#888"
    )
    plt.title("tree")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


def plot_cmnist(data):
    G = to_networkx(data, to_undirected=True)

    # coordinates are last 2 features
    coords = data.x[:, -2:].cpu().numpy()

    # first 3 features are RGB color
    colors = data.x[:, :3].cpu().numpy()

    pos = {i: coords[i] for i in range(coords.shape[0])}

    plt.figure(figsize=(6, 6))
    nx.draw(
        G,
        pos,
        node_color=colors,
        node_size=80,
        with_labels=False
    )
    plt.title(f"Label: {data.y.item()}")
    plt.gca().invert_yaxis()
    plt.show()


def saliency_to_probs_single(node_imp: torch.Tensor, tau: float = 0.25):
    return torch.softmax(node_imp / tau, dim=0)


def soft_target_from_mask_single(mask: torch.Tensor, eps: float = 1e-9):
    mask = mask.bool()
    q = torch.zeros_like(mask, dtype=torch.float)
    q[mask] = 1.0 / mask.sum()
    return q.clamp_min(eps)


def saliency_grad_diff(model, batch, epoch=None):
    # model.eval()
    # if batch.x.shape[1]>2:
    #     batch.x = batch.x[:, 3:]
    x = batch.x.clone().requires_grad_(True)

    logits = model(x, batch.edge_index, batch.batch)

    log_probs = torch.log_softmax(logits, dim=-1)
    scalar = log_probs.sum()
    grads = torch.autograd.grad(
        scalar, x,
        create_graph=True,
        retain_graph=True
    )[0]
    # model.train()
    #
    # feature_grad = grads.pow(2).mean(axis=0).detach().numpy()
    #
    # plt.bar(range(len(feature_grad)), feature_grad)
    # plt.xlabel("Feature")
    # plt.ylabel("Average Gradient")
    # plt.title("Average Gradient per Feature")
    # plt.show()
    # plot_cmnist(batch.to_data_list()[0])

    # per_graph_feat_imp = []
    #
    # for g_id in range(batch.num_graphs):
    #     g_mask = (batch.batch == g_id)
    #     g_grads = grads[g_mask]
    #     per_graph_feat_imp.append(g_grads.pow(2).mean(dim=0))
    #
    # per_graph_feat_imp = torch.stack(per_graph_feat_imp)
    #
    # mean_imp = per_graph_feat_imp.mean(dim=0).detach().numpy()
    # std_imp = per_graph_feat_imp.std(dim=0).detach().numpy()
    #
    # plt.bar(range(len(mean_imp)), mean_imp, yerr=std_imp)
    # plt.title("Feature importance (mean ± std)")
    # plt.show()
    #
    # dominant_feat = per_graph_feat_imp.argmax(dim=1)
    # plt.hist(dominant_feat.cpu().numpy(), bins=range(grads.shape[1] + 1))
    # plt.title("Dominant feature per graph")
    # plt.show()
    #
    # g_id = (dominant_feat == 1).nonzero(as_tuple=True)[0][0].item()
    # g_mask = (batch.batch == g_id)
    # g_grads = grads[g_mask]
    # node_imp = g_grads[:, 1].pow(2)
    # gg = batch.to_data_list()[g_id]
    # conf_mask = (gg.sp_order == 0) | (gg.sp_order == gg.sp_order.max())
    # digit_mask = gg.node_label.bool()
    #
    # print("conf nodes mean:", node_imp[conf_mask].mean().item())
    # print("other nodes mean:", node_imp[~conf_mask].mean().item())
    #
    # plot_node_importance(
    #     gg,
    #     digit_mask.nonzero(as_tuple=True)[0],
    #     conf_mask.nonzero(as_tuple=True)[0],
    #     node_imp,
    #     title=f"Graph {g_id} (feature 1 dominant)"
    # )

    node_imp = (grads.pow(2).sum(dim=1) + 1e-9).sqrt()# [N], raw real-valued importance
    # import pdb;pdb.set_trace()
    # conf_mask = (batch.sp_order == 0) | (batch.sp_order == batch.sp_order.max())
    # digit_mask = batch.node_label.bool()
    # graph_mask = (batch.batch == 0)
    # conf_mask_0 = conf_mask & graph_mask
    # digit_mask_0 = digit_mask & graph_mask
    # plot_node_importance(batch.to_data_list()[0], digit_mask_0.nonzero(as_tuple=True)[0], conf_mask_0.nonzero(as_tuple=True)[0], grads[:len(batch.to_data_list()[0].sp_order),1].pow(2),
    #                      title="Node Importance")

    # # abs grads
    # g = grads.abs()
    #
    # # RGB part (features 0–2)
    # node_grad_rgb = g[:, :3].sum(dim=1)
    #
    # # rest of features (4:)
    # node_grad_rest = g[:, 4:].sum(dim=1)
    #
    # print("RGB conf:", node_grad_rgb[conf_mask].mean())
    # print("RGB digit:", node_grad_rgb[digit_mask].mean())
    #
    # print("REST conf:", node_grad_rest[conf_mask].mean())
    # print("REST digit:", node_grad_rest[digit_mask].mean())
    # print("--------------------------------------------")
    # #

    aucs = []
    node_imp2 = node_imp.clone()
    #
    if len(batch.batch.unique()) > 5 and model.training:
        available_g = np.random.choice(batch.batch.unique().cpu(), 5, replace=False)
    elif not model.training and epoch is not None:
        available_g = np.random.choice(batch.batch.unique().cpu(), 100, replace=False)
    else:
        available_g = batch.batch.unique()
    if epoch is not None:
        if epoch % 5 == 0:
            for g_id in available_g:
                m = (batch.batch == g_id)  # nodes of this graph
                motif_mask_g = batch.motif_node_mask[m].bool() if batch.x.shape[1] == 7 else batch.node_label[m].bool()

                auc = roc_auc_score(motif_mask_g.cpu().numpy().astype(np.int32), node_imp[m].cpu().detach().numpy().astype(np.float32))
                aucs.append(auc)
    else:
        aucs = None

    saliency = grads.pow(2)#.abs()
    auc_value = float(np.mean(aucs)) if aucs is not None and len(aucs)>0 else None

    return node_imp2, saliency, auc_value


@torch.no_grad()
def compute_plausibility(expl, batch):
    hits = []
    aucs = []
    node_imp2 = expl.clone()

    for g_id in batch.batch.unique():
        m = (batch.batch == g_id)  # nodes of this graph
        motif_mask_g = batch.motif_node_mask[m].bool() if batch.x.shape[1] == 7 else batch.node_label[m].bool()
        auc = roc_auc_score(motif_mask_g.cpu().numpy().astype(np.int32), expl[m].cpu().detach().numpy().astype(np.float32))
        aucs.append(auc)
    return float(np.mean(aucs))


@torch.no_grad()
def uncertainty_scores_logits(model, pool, device, method="entropy"):
    model.eval()

    all_scores = []
    all_ids = []

    for batch in pool:
        batch = batch.to(device)

        logits = model(batch.x, batch.edge_index, batch.batch)
        probs = F.softmax(logits, dim=-1)

        if method == "least-confidence":
            scores = 1.0 - probs.max(dim=-1).values

        elif method == "margin":
            top2 = probs.topk(2, dim=-1).values
            scores = 1.0 - (top2[:, 0] - top2[:, 1])

        elif method == "entropy":
            eps = 1e-12
            scores = -(probs * (probs + eps).log()).sum(dim=-1)

        elif method == 'random':
            raw_scores = torch.ones(probs.size(0))
            noise = torch.rand_like(raw_scores) * 1e-3
            scores = raw_scores + noise

        all_scores.append(scores.cpu())
        all_ids.append(batch.idx.cpu())

    all_scores = torch.cat(all_scores)
    all_ids = torch.cat(all_ids)

    return all_scores


def uncertainty_scores_e(model, pool, device, method="entropy"):
    # model.eval()

    all_scores = []
    all_ids = []

    for batch in pool:
        batch = batch.to(device)

        _, sal, _ = saliency_grad_diff(model, batch, epoch=None)

        node_imp_raw = sal.sum(dim=1)
        num_graphs = int(batch.y.size(0))
        node_imp = softmax(node_imp_raw, batch.batch, num_nodes=num_graphs)

        if method == "margin":
            top2 = node_imp_raw.topk(2, dim=-1).values
            scores = top2[0] - top2[1]

        elif method == "entropy":
            eps = 1e-12
            scores = []
            for g in range(num_graphs):
                mask = (batch.batch == g)
                p = node_imp[mask]
                N = len(p)

                H = -(p * (p + eps).log()).sum()
                H_norm = H / torch.log(torch.tensor(float(N), device=p.device))
                scores.append(H_norm)

        elif method == 'random':
            raw_scores = torch.ones(num_graphs)
            noise = torch.rand_like(raw_scores) * 1e-3
            scores = raw_scores + noise

        if method == 'entropy':
            all_scores.append(torch.stack(scores).cpu())
        else:
            all_scores.append((scores.cpu()))
        all_ids.append(batch.idx.cpu())

    all_scores = torch.cat(all_scores)
    all_ids = torch.cat(all_ids)

    return all_scores


@torch.no_grad()
def select_topk(pool, scores, k):
    topk_idx = torch.topk(scores, k, largest=False).indices
    print(torch.topk(scores, k, largest=False))
    print(f'max: {scores.max()}, min: {scores.min()}, mean: {scores.mean()}')
    selected_ids = [pool[i] for i in topk_idx.tolist()]
    return selected_ids
