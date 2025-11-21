import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
from utils import *
from models import *
# import wandb
from tap import Tap

torch.set_num_threads(6)


SEED = 42
DEVICE = "cuda:0"
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


class Arguments(Tap):
    epochs: int = 500
    runs: int = 1
    lr: float = 1e-4
    supervision_rate: float = 0.1
    lam_ce: float = 1.
    lam_expl: float = 0.1
    mode: str = 'passive-exp' # or 'no-supervision'
    log_wandb: bool = True


def run_exp(args: Arguments):
    # wandb.init(project='xilgraph',
    #            entity='xilgraph',
    #            mode='online' if args.log_wandb else 'disabled',
    #            config=args.as_dict())
    # Generate a tree for each class
    trees = generate_trees(n_tree, tree_colors)
    graphs_by_splits = {}
    for split, n in n_splits.items():
        graphs = []

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

    # model = GCN().to(DEVICE)
    # model = GAT().to(DEVICE)
    # model = GIN().to(DEVICE)
    model = SAGE().to(DEVICE)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    criterion_bce = nn.BCELoss()
    if args.mode == 'no-supervision':
        for epoch in range(1, args.epochs + 1):
            tr_loss, tr_acc = run_epoch(model, train_loader, opt, criterion, train=True)
            val_loss, val_acc = run_epoch(model, val_loader, opt, criterion, train=False)
            if epoch % 5 == 0 or epoch == 1:
                print(f"Epoch {epoch:02d} | "
                      f"train loss {tr_loss:.3f} acc {tr_acc:.3f} | "
                      f"val loss {val_loss:.3f} acc {val_acc:.3f}")

        total_val_acc = val_acc
        # Final test
        test_loss, test_acc = run_epoch(model, test_loader, opt, criterion, train=False)
        total_test_acc = test_acc
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

    elif args.mode == 'passive-exp':
        for epoch in range(1, args.epochs + 1):
            model.train()

            correct = 0.
            total = 0.
            cnt = 0.
            total_loss = 0.
            total_expl = 0.
            average_n_hit = 0.
            average_aucs = 0.

            for batch in train_loader:
                batch = batch.to(DEVICE)
                out = model(batch.x, batch.edge_index, batch.batch)
                correct += (out.argmax(dim=-1) == batch.y.view(-1)).sum().item()
                total += batch.y.view(-1).size(0)
                ce_loss = criterion(out, batch.y.view(-1))

                expl_loss = torch.tensor(0.0, device=DEVICE)

                chosen = (torch.rand(batch.y.view(-1).size(0), device=DEVICE) < args.supervision_rate)
                if chosen.any():
                    # model.eval()

                    cnt += 1
                    gt_mask = batch.motif_node_mask.to(DEVICE).float()
                    _, sal, n_hit, aucs = saliency_grad_diff(model, batch)

                    node_imp = sal.sum(dim=1)

                    # positive mask: want high saliency
                    pos_loss = -torch.mean(node_imp * gt_mask)
                    # Negative mask: want low saliency
                    neg_loss = torch.mean(node_imp * (1 - gt_mask))

                    expl_loss = pos_loss + neg_loss

                    expl_loss = torch.clamp(expl_loss, min=-100, max=100)

                    # model.train()
                    average_n_hit += n_hit
                    average_aucs += aucs

                    # node_sel = chosen[batch.batch].float()
                    # if node_sel.sum() > 0:
                    #     expl_loss = F.binary_cross_entropy(
                    #         node_imp,
                    #         gt_mask,
                    #         weight=node_sel,
                    #         reduction="sum",
                    #     ) / node_sel.sum()

                loss = args.lam_ce * ce_loss + args.lam_expl * expl_loss
                loss.backward()
                # for n , p in model.named_parameters():
                #     print(n,p.grad.norm())
                opt.step()
                opt.zero_grad()
                total_loss += float(loss.detach())
                total_expl += float(expl_loss.detach())

        #
        # for epoch in range(1, args.epochs + 1):
        #     total_loss = 0
        #     total_expl = 0
        #     model.train()
        #     correct = 0.
        #     total = 0.
        #     cnt = 0.
        #     average_n_hit = 0.
        #     for g in graphs_by_splits['train']:
        #         g = g.to(DEVICE)
        #         gt_mask = torch.zeros(g.num_nodes)
        #         out = model(g.x, g.edge_index, g.batch)
        #         correct += (out.argmax(dim=-1) == g.y.view(-1)).sum().item()
        #         total += 1
        #         ce_loss = criterion(out, g.y.view(-1))
        #
        #         expl_loss = 0.0
        #         # lam_base = 1
        #         # lam = 0
        #         if torch.rand(()) < args.supervision_rate and hasattr(g, "motif_node_ids"):
        #             # get Captum explanation for this graph
        #             model.eval()
        #             # node_imp, n_hit, _ = captum_explain_graphs(model, g, num_samples=1, method="IntegratedGradients")
        #             # if epoch % 5 == 0: # or epoch == 1:
        #             #     plot_node_importance(g, g.motif_node_ids, node_imp, title="Captum Node Importance")
        #
        #             model.train()
        #             cnt += 1
        #             gt_mask[g.motif_node_ids] = 1.
        #
        #             # explanation loss
        #             pos = gt_mask.sum().clamp_min(1.0)
        #             neg = (1 - gt_mask).sum().clamp_min(1.0)
        #             pos_weight = (neg / pos)  # >1 if positives are rare
        #             w = torch.ones_like(gt_mask)
        #             w[gt_mask == 1] = pos_weight  # emphasize positives
        #
        #             node_imp, sal, n_hit = saliency_grad_diff(model, g)
        #
        #             average_n_hit += n_hit
        #
        #             # turn node_imp into logits (zero-mean, scaled); this keeps gradient stable
        #             #imp_logits = (node_imp - node_imp.mean().detach()) / (node_imp.std().detach() + 1e-9)
        #
        #             # expl_loss = F.binary_cross_entropy_with_logits(imp_logits, gt_mask, weight=w)
        #
        #             expl_loss = F.binary_cross_entropy(node_imp, gt_mask) #, weight=w)
        #             # p = saliency_to_probs_single(node_imp, tau=0.25)
        #             # q = soft_target_from_mask_single(gt_mask)
        #
        #             # expl_loss = F.kl_div(p.log(), q, reduction="batchmean")
        #             # lam = lam_base * (1 + epoch / 10)
        #
        #         loss = args.lam_ce * ce_loss + args.lam_expl * expl_loss
        #         loss.backward()
        #         # for n , p in model.named_parameters():
        #         #     print(n,p.grad.norm())
        #         opt.step()
        #         opt.zero_grad()
        #         total_loss += float(loss.detach())
        #         total_expl += float(expl_loss)
       
            tr_acc = correct / max(total, 1)
            total_loss = total_loss / max(len(train_loader), 1)
            total_expl = total_expl / max(len(train_loader), 1)
            tr_average_n_hit = average_n_hit / cnt if cnt > 0 else 0
            average_aucs = average_aucs / cnt if cnt > 0 else 0

            val_loss, val_acc = run_epoch(model, val_loader, opt, criterion, train=False, device=DEVICE)
            val_batch = Batch.from_data_list(val_set).to(DEVICE)
            _, _, val_average_n_hit, val_aucs = saliency_grad_diff(model, val_batch)

            # _, val_average_n_hit, average_e_hit = captum_explain_graphs(model, val_set, num_samples=len(val_set),
            #                                                         method="IntegratedGradients")

            log_dict = {'epoch': epoch,
                        'total_loss_tr': total_loss,
                        'expl_loss': total_expl,
                        'loss_val': val_loss,
                        'acc_tr': tr_acc,
                        'acc_val': val_acc,
                        'tr_n_hit': tr_average_n_hit,
                        'val_n_hit': val_average_n_hit,
                        'train_auc': average_aucs,
                        'val_auc': val_aucs}
            # wandb.log(log_dict)

            if epoch % 1 == 0 or epoch == 1:
                print(f"Epoch {epoch:02d} | "
                      f"train loss {total_loss:.3f} expl loss {total_expl:.5f} acc {tr_acc:.3f} | val loss "
                      f"{val_loss:.3f} val acc {val_acc:.3f}")
                print(f"train average motif hit: {tr_average_n_hit:.3f} | val average motif hit {val_average_n_hit:.3f} | train AUC {average_aucs:.3f}")

        total_val_acc = val_acc
        test_loss, test_acc = run_epoch(model, test_loader, opt, criterion, train=False, device=DEVICE)
        total_test_acc = test_acc
        print(f"Test  | loss {test_loss:.3f} acc {test_acc:.3f}")
    return total_val_acc, total_test_acc


args = Arguments(explicit_bool=True).parse_args()
total_val_acc = torch.empty(args.runs)
total_test_acc = torch.empty(args.runs)

for run in range(args.runs):
    val_acc, test_acc = run_exp(args)
    total_val_acc[run] = val_acc
    total_test_acc[run] = test_acc

print(f'Average val acc: {total_val_acc.mean() * 100:.2f} ± {total_val_acc.std() * 100:.2f}')
print(f'Average test acc: {total_test_acc.mean() * 100:.2f} ± {total_test_acc.std() * 100:.2f}')
