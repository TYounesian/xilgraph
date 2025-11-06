import torch
from torch_geometric.loader import DataLoader
from utils import *
from models import *

SEED = 42
DEVICE = "cpu"
n_tree = 6
NUM_GRAPHS = 1000
N_NODES = 50          # base graph size
P_EDGE = 0.05         # Erdős–Rényi edge probability
n_splits = {'train': int(0.7*NUM_GRAPHS), 'val': int(0.15*NUM_GRAPHS), 'test': int(0.15*NUM_GRAPHS)}
CID = {"red": 0, "blue": 1, "green": 2, "yellow": 3, "orange": 4, "purple": 5, "cyan": 6}
tree_colors = list(CID.values())[0:3]
graph_colors = list(CID.values())[0:-2]
# class target colors
target_colors = random.sample(graph_colors[-2:], k=2)

EPOCHS = 10
runs = 1

mode = 'passive-exp' # or 'no-supervision'

total_val_acc = torch.empty(runs)
total_test_acc = torch.empty(runs)

# Generate a tree for each class
trees = generate_trees(n_tree, tree_colors)
graphs_by_splits = {}
for split, n in n_splits.items():
    graphs = []
    motif_ex_count = 0

    for _ in range(n):
        G = generate_and_check(trees, N_NODES, P_EDGE, graph_colors)
        g = make_graph(trees, G, CID, target_colors, split)
        # plot_g_tree(g, trees, CID)
        graphs.append(g)

    graphs_by_splits[split] = graphs
    # print(f'Percentage of graphs that already have at least one of the motifs: {motif_ex_count/n*100}')

torch.manual_seed(SEED)

train_set = graphs_by_splits['train']
val_set = graphs_by_splits['val']
test_set = graphs_by_splits['test']

train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
val_loader = DataLoader(val_set, batch_size=16, shuffle=False)
test_loader = DataLoader(test_set, batch_size=16, shuffle=False)

for run in range(runs):
    model = GCN().to(DEVICE)
    # model = GAT().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()
    if mode == 'no-supervision':
        for epoch in range(1, EPOCHS + 1):
            tr_loss, tr_acc = run_epoch(model, train_loader, opt, criterion, train=True)
            val_loss, val_acc = run_epoch(model, val_loader, opt, criterion, train=False)
            if epoch % 5 == 0 or epoch == 1:
                print(f"Epoch {epoch:02d} | "
                      f"train loss {tr_loss:.3f} acc {tr_acc:.3f} | "
                      f"val loss {val_loss:.3f} acc {val_acc:.3f}")

        total_val_acc[run] = val_acc
        # Final test
        test_loss, test_acc = run_epoch(model, test_loader, opt, criterion, train=False)
        total_test_acc[run] = test_acc
        print(f"Test  | loss {test_loss:.3f} acc {test_acc:.3f}")

        average_f1, average_r = grad_explainer(model, graphs_by_splits, trees)
        print(f'Average Train F1: {average_f1}')
        print(f'Average Train Recall: {average_r}')
        # print("Top-10 node indices:", torch.topk(node_imp, k=10).indices.tolist())
        # print("Motif node indices:", motif_nodes.tolist())

        # call it for each split (or just train/test)
        _, average_n_hit, average_e_hit = captum_explain_graphs(model, train_set, num_samples=len(train_set),
                                                             method="IntegratedGradients")
        # print(f"Average attach node hit@top20%= {average_n_hit:.3f}")
        print(f"Average motif node hit@top20% = {average_n_hit:.3f}, "
              f"Average motif edge hit@top20% = {average_e_hit:.3f}")

        # captum_explain_graphs(model, val_set, num_samples=len(val_set), method="IntegratedGradients")
        # captum_explain_graphs(model, test_set, num_samples=len(test_set), method="IntegratedGradients")
    elif mode == 'passive-exp':
        for epoch in range(1, EPOCHS + 1):
            total_loss = 0
            total_ce = 0
            model.train()
            correct = 0.
            total = 0.
            cnt = 0.
            average_n_hit = 0.
            for g in graphs_by_splits['train']:
                g = g.to(DEVICE)
                gt_mask = torch.zeros(g.num_nodes)
                out = model(g.x, g.edge_index, g.batch)
                correct += (out.argmax(dim=-1) == g.y.view(-1)).sum().item()
                total += 1
                ce_loss = criterion(out, g.y.view(-1))

                expl_loss = 0.0
                if torch.rand(()) < 0.1 and hasattr(g, "motif_node_ids"):
                    # get Captum explanation for this graph
                    node_imp, n_hit, _ = captum_explain_graphs(model, g, num_samples=1, method="IntegratedGradients")
                    # if epoch % 5 == 0: # or epoch == 1:
                    #     plot_node_importance(g, g.motif_node_ids, node_imp, title="Captum Node Importance")

                    average_n_hit += n_hit
                    cnt += 1
                    # ground truth mask (0/1)
                    gt_mask[g.motif_node_ids] = 1.

                    # explanation loss (BCE style)
                    expl_loss = F.binary_cross_entropy(node_imp, gt_mask)

                loss = ce_loss + expl_loss
                loss.backward()
                opt.step()
                opt.zero_grad()
                total_loss += float(loss)
                total_ce += float(0.8 * expl_loss)
            tr_acc = correct / max(total, 1)
            total_loss = total_loss / max(total, 1)
            total_ce = total_ce / total
            tr_average_n_hit = average_n_hit / cnt

            val_loss, val_acc = run_epoch(model, val_loader, opt, criterion, train=False)
            _, val_average_n_hit, average_e_hit = captum_explain_graphs(model, val_set, num_samples=len(val_set),
                                                                    method="IntegratedGradients")

            if epoch % 5 == 0 or epoch == 1:
                print(f"Epoch {epoch:02d} | "
                      f"train loss {total_loss:.3f} expl loss {total_ce:.3f} acc {tr_acc:.3f} | val loss "
                      f"{val_loss:.3f} val acc {val_acc:.3f}")
                print(f"train average motif hit: {tr_average_n_hit}| val average motif hit {val_average_n_hit}")

        total_val_acc[run] = val_acc

        test_loss, test_acc = run_epoch(model, test_loader, opt, criterion, train=False)
        total_test_acc[run] = test_acc
        print(f"Test  | loss {test_loss:.3f} acc {test_acc:.3f}")

print(f'Average val acc: {total_val_acc.mean() * 100:.2f} ± {total_val_acc.std() * 100:.2f}')
print(f'Average test acc: {total_test_acc.mean() * 100:.2f} ± {total_test_acc.std() * 100:.2f}')
