import pdb

import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
from utils import *
from models import *
from se_models import *
import wandb
from tap import Tap
import math
from datasets import CPatchMNIST
from torch_geometric.data import InMemoryDataset
import pdb

torch.set_num_threads(6)


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
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


class Arguments(Tap):
    epochs: int = 200
    runs: int = 1
    lr: float = 1e-4
    supervision_rate: float = 1
    lam_ce: float = 1.
    lam_expl: float = 1
    mode: str = 'passive-exp' # or 'no-supervision' or 'active_exp'
    log_wandb: bool = True
    model: str = 'gcn'
    explainer: str = 'post' # post | ante
    active: str = 'least-confidence'
    per_round: int = 5
    rounds: int = 10
    dataset: str = 'synth'
    only_neg: bool = False
    id_test: bool = False
    binary: bool = False


def run_exp(args: Arguments):
    wandb.init(project='xilgraph',
               entity='xilgraph',
               mode='online' if args.log_wandb else 'disabled',
               config=args.as_dict())
    # Generate a tree for each class
    torch.manual_seed(SEED)
    random.seed(SEED)
    if args.dataset == 'synth':
        trees = generate_trees(n_tree, tree_colors)
        graphs_by_splits = {}
        for split, n in n_splits.items():
            graphs = []

            for _ in range(n):
                G = generate_and_check(trees, N_NODES, P_EDGE, graph_colors)
                confounder_flag = False
                if split == 'train':
                    confounder_flag = True
                g = make_graph(trees, G, CID, target_colors, split, confounder_flag)
                # plot_g_tree(g, trees, CID)
                graphs.append(g)

            graphs_by_splits[split] = graphs
            # print(f'Percentage of graphs that already have at least one of the motifs: {motif_ex_count/n*100}')

        train_set = graphs_by_splits['train']
        val_set = graphs_by_splits['val']
        test_set = graphs_by_splits['test']

        in_dim = train_set[0].x.shape[1]
        out_dim = 2
        batch_size = 16

    elif args.dataset == 'cmnist':

        dataset = CPatchMNIST.load(dataset_root="./data")

        class FilteredDataset(InMemoryDataset):
            def __init__(self, data_list):
                super().__init__(root='')  # dummy root
                self.data, self.slices = self.collate(data_list)

        def keep_zero_one(dataset):
            data_list = [d for d in dataset if d.y.item() in (0, 1)]
            return FilteredDataset(data_list)
        if args.binary:
            train_set = keep_zero_one(dataset["train"])
            val_set = keep_zero_one(dataset["val"])
            if args.id_test:
                test_set = keep_zero_one(dataset["id_test"])
            else:
                test_set = keep_zero_one(dataset["test"])
        else:
            train_set = dataset['train']
            val_set = dataset['val']

            if args.id_test:
                test_set = dataset["id_test"]
            else:
                test_set = dataset["test"]

        in_dim = train_set[0].x.shape[1]
        out_dim = int(train_set.data.y.max()) + 1
        batch_size = 256

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    if args.explainer == "post":
        if args.model == 'gcn':
            model = GCN(in_dim=in_dim, out_dim=out_dim).to(DEVICE)
        elif args.model == 'gat':
            model = GAT(in_dim=in_dim, out_dim=out_dim).to(DEVICE)
        elif args.model == 'gat2':
            model = GATv2(in_dim=in_dim, out_dim=out_dim).to(DEVICE)
        elif args.model == 'gin':
            model = GIN(in_dim=in_dim, out_dim=out_dim).to(DEVICE)
        elif args.model == 'sage':
            model = SAGE(in_dim=in_dim, out_dim=out_dim).to(DEVICE)
    elif args.explainer == "ante":
        if args.model == 'gin':
            model = SEGIN(in_dim=in_dim, out_dim=out_dim, disable_expl=args.lam_expl == 0.0).to(DEVICE)
        elif args.model == 'sage':
            model = SESAGE(in_dim=in_dim, out_dim=out_dim, disable_expl=args.lam_expl == 0.0).to(DEVICE)
        elif args.model == 'gcn':
            model = SEGCN(in_dim=in_dim, out_dim=out_dim, disable_expl=args.lam_expl == 0.0).to(DEVICE)
        elif args.model == 'gat':
            model = SEGAT(in_dim=in_dim, out_dim=out_dim, disable_expl=args.lam_expl == 0.0).to(DEVICE)
        else:
            raise NotImplementedError(f"{args.model} not yet implemented for {args.explainer} modality")
    else:
        raise NotImplementedError(f"{args.explainer} not yet implemented")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    if args.dataset == 'synth':
        for i, data in enumerate(train_set):
            data.idx = torch.tensor([i])

    if args.mode == 'no-supervision':
        for epoch in range(1, args.epochs + 1):
            tr_loss, tr_acc = run_epoch(model, train_loader, opt, criterion, epoch, train=True, device=DEVICE)
            val_loss, val_acc = run_epoch(model, val_loader, opt, criterion, epoch, train=False, device=DEVICE)
            test_loss, test_acc = run_epoch(model, test_loader, opt, criterion, epoch, train=False, device=DEVICE)
            log_dict = {'epoch': epoch,
                        'total_loss_tr': tr_loss,
                        'loss_val': val_loss,
                        'acc_tr': tr_acc,
                        'acc_val': val_acc,
                        'acc test': test_acc}
            wandb.log(log_dict)
            if epoch % 2 == 0 or epoch == 1:
                print(f"Epoch {epoch:02d} | "
                      f"train loss {tr_loss:.3f} acc {tr_acc:.3f} | "
                      f"val loss {val_loss:.3f} acc {val_acc:.3f} | "
                      f"test acc {test_acc:.3f}")

        total_val_acc = val_acc

        # Final test
        test_loss, test_acc = run_epoch(model, test_loader, opt, criterion, epoch, train=False, device=DEVICE)
        total_test_acc = test_acc
        print(f"Test  | loss {test_loss:.3f} acc {test_acc:.3f}")

    elif args.mode == 'passive-exp':
        for epoch in range(1, args.epochs + 1):
            model.train()

            correct = 0.
            total = 0.
            cnt = 0.
            total_loss = 0.
            total_expl = 0.
            total_ce = 0.
            average_aucs = 0.
            cnttt = 0

            pos_sum, pos_count = 0.0, 0
            neg_sum, neg_count = 0.0, 0
            conf_sum, conf_count = 0.0, 0
            for batch in train_loader:
                batch = batch.to(DEVICE)

                batch_size = batch.y.view(-1).size(0)
                out = model(batch.x, batch.edge_index, batch.batch)

                if args.explainer == "ante":
                    expl_attn_logit, out = out # separate explanation from target predictions

                correct += (out.argmax(dim=-1) == batch.y.view(-1)).sum().item()
                total += batch.y.view(-1).size(0)
                ce_loss = criterion(out, batch.y.view(-1))

                expl_loss = torch.tensor(0.0, device=DEVICE)

                chosen_mask = torch.zeros(batch_size, device=DEVICE)

                chosen = (torch.rand(batch.y.view(-1).size(0), device=DEVICE) < args.supervision_rate)
                chosen_mask[chosen] = 1.0

                if args.supervision_rate > 0:
                    cnt += 1
                    node_mask = chosen_mask[batch.batch].bool()

                    graph_indices = chosen_mask.nonzero(as_tuple=True)[0]  # graph ids in this batch
                    sub_batch = Batch.from_data_list(batch.index_select(graph_indices))

                    gt_mask = batch.motif_node_mask[node_mask].to(DEVICE).float() if args.dataset == 'synth' else batch.node_label.bool()[node_mask].to(DEVICE).float()

                    if args.explainer == "post":
                        _, sal, aucs = saliency_grad_diff(model, sub_batch, epoch)

                        node_imp = sal.sum(dim=1)

                        # positive mask: want high saliency
                        pos_loss = -node_imp[gt_mask.bool()].sum()
                        # Negative mask: want low saliency
                        neg_loss = node_imp[~gt_mask.bool()].sum()

                        graphs_in_batch = batch.to_data_list()
                        for i in range(batch_size):
                            g0 = graphs_in_batch[i]
                            mask = batch.batch == i
                            auc_g = roc_auc_score(g0.node_label.bool().detach().numpy(),
                                                  node_imp[mask].detach().numpy())
                            # print(f'label {g0.y}, out {out[i,:].argmax()}, auc {auc_g}')
                            average_aucs += auc_g

                        #
                        # if epoch % 2 == 0:
                        #     # Positive nodes
                        #     pos_vals = node_imp[gt_mask.bool()]
                        #     pos_sum += pos_vals.sum().item()
                        #     pos_count += pos_vals.numel()
                        #     #
                            # # Negative nodes
                            # neg_vals = node_imp[~gt_mask.bool()]
                            # neg_sum += neg_vals.sum().item()
                            # neg_count += neg_vals.numel()
                            #
                            # if args.dataset == 'synth':
                            #     conf_vals = node_imp[batch.conf_id]
                            # else:
                            #     for g in graph_indices:
                            #         mask = (batch.batch == g)
                            #
                            #         local_sp = batch.sp_order[mask]
                            #         conf_id = ((local_sp == 0) | (local_sp == local_sp.max())).nonzero(as_tuple=True)[0]
                            #
                            #         node_imp_g = node_imp[mask]
                            #         conf_vals = node_imp_g[conf_id]
                            #
                            # conf_sum += conf_vals.sum().item()
                            # conf_count += conf_vals.numel()
                            # print(f"batch {cnt} positive sum {pos_vals.sum().item():.1f} negative sum {neg_vals.sum().item():.1f} conf sum {conf_vals.sum().item():.1f}")
                            #
                            # if (epoch == 2 and cnt <4) or (epoch > 7 and epoch % 5 ==0 and cnt <4):
                            #     graphs_in_batch = batch.to_data_list()
                            #     g0 = graphs_in_batch[12]
                            #     mask = batch.batch == 12
                            #     auc_g = roc_auc_score(g0.node_label.bool().detach().numpy(), node_imp[mask].detach().numpy())
                            #     print(f'label {g0.y}, out {out[12,:].argmax()}, auc {auc_g}')

                                # motif_node_ids = torch.arange(sum(batch.batch==12))[batch.node_label[batch.batch==12].bool()]
                                # conf_id_g = ((batch.sp_order[batch.batch==12] == 0) | (batch.sp_order[batch.batch==12] == batch.sp_order[batch.batch==12].max())).nonzero(as_tuple=True)[0]
                                # plot_node_importance(g0, motif_node_ids, conf_id_g, node_imp[batch.batch==12],
                                #                 title="Node Importance")
                            # plot_g_tree(g0, trees, CID, node_imp[0:len(g0.y_color)])
                            # print(f'max node_imp {node_imp[0:len(g0.y_color)].max()},  and total average {node_imp[0:len(g0.y_color)].mean()}')
                        if args.only_neg:
                            expl_loss = neg_loss / (node_imp.sum() + 1e-6)
                        else:
                            expl_loss = neg_loss / (node_imp.sum() + 1e-6) + pos_loss / (node_imp.sum() + 1e-6)

                        log_dict = {
                            'batch_expl_loss': expl_loss,
                            'p_loss': pos_loss,
                            'n_loss': neg_loss
                        }
                        wandb.log(log_dict)
                    else:
                        expl_loss = F.binary_cross_entropy_with_logits(expl_attn_logit, gt_mask)
                        if epoch % 2 == 0:
                            aucs = compute_plausibility(expl_attn_logit, sub_batch)
                        else:
                            aucs = None

                    if aucs is not None:
                        average_aucs += aucs

                reg = 0 #(node_imp ** 2).mean()

                expl_loss = torch.clamp(expl_loss, min=-1000, max=1000)
                # if epoch <20:
                #     lam_ce = 0.
                # else:
                #     lam_ce = args.lam_ce
                loss = args.lam_ce * ce_loss + args.lam_expl * expl_loss #+ 1e-5 * sum(p.pow(2).sum() for p in model.parameters()) #+ 0.005*reg

                opt.zero_grad()
                loss.backward()
                opt.step()
                total_loss += float(loss.detach())
                total_expl += float(expl_loss.detach())
                total_ce += float(ce_loss.detach())

            tr_acc = correct / max(total, 1)
            total_loss = total_loss / max(len(train_loader), 1)
            total_expl = total_expl / max(len(train_loader), 1)
            total_ce = total_ce / max(len(train_loader), 1)
            average_aucs = average_aucs / max(len(train_loader), 1)

            pos_avg = pos_sum / pos_count if pos_count > 0 else 0
            neg_avg = neg_sum / neg_count if neg_count > 0 else 0
            conf_avg = conf_sum / conf_count if conf_count > 0 else 0

            if epoch % 2 == 0:
                # print(f"Epoch {epoch}")
                # print("positives average:", pos_avg)
                # print("negatives average:", neg_avg)
                # print("confounder average:", conf_avg)

                log_dict = {
                    'positives average': pos_avg,
                    'negatives average': neg_avg,
                    'confounder average': conf_avg
                }
                wandb.log(log_dict)

            model.eval()
            val_loss, val_acc = run_epoch(model, val_loader, opt, criterion, epoch, train=False, device=DEVICE)
            test_loss, test_acc = run_epoch(model, test_loader, opt, criterion, epoch, train=False, device=DEVICE)
            val_batch = Batch.from_data_list(val_set).to(DEVICE)
            if args.explainer == "post":
                _, _, val_aucs = saliency_grad_diff(model, val_batch, epoch)
            else:
                expl_attn_logit, out = model(val_batch.x, val_batch.edge_index, val_batch.batch)
                val_aucs = compute_plausibility(expl_attn_logit, val_batch)

            log_dict = {'epoch': epoch,
                        'total_loss_tr': total_loss,
                        'expl_loss': total_expl,
                        'ce_loss': total_ce,
                        'loss_val': val_loss,
                        'acc_tr': tr_acc,
                        'acc_val': val_acc,
                        'acc test': test_acc}

            wandb.log(log_dict)
            if val_aucs is not None:
                wandb.log({"val_auc": val_aucs})
            if average_aucs > 0:
                wandb.log({"train_auc": average_aucs})

            if epoch % 2 == 0:
                print(f"Epoch {epoch:02d} | "
                      f"train loss {total_loss:.3f} expl loss {total_expl:.5f} reg {reg:.5f} acc {tr_acc:.3f} | val loss "
                      f"{val_loss:.3f} val acc {val_acc:.3f} | test acc : {test_acc}")
                if val_aucs is None:
                    val_aucs = 0
                print(f"train AUC {average_aucs:.3f} | val AUC {val_aucs:.3f}")

        total_val_acc = val_acc
        test_loss, test_acc = run_epoch(model, test_loader, opt, criterion, epoch, train=False, device=DEVICE)
        total_test_acc = test_acc
        print(f"Test  | loss {test_loss:.3f} acc {test_acc:.3f}")
    elif args.mode == 'active-exp':
        explained_idx = set()  # graphs with explanation labels
        all_idx = {data.idx.item() for data in train_set}
        # explained_idx.update(random.sample(all_idx, len(train_set)))  # If we want to pre-train expl

        per_round = args.per_round
        for rounds in range(args.rounds):
            # If re-start is needed in every round
            # torch.manual_seed(SEED)
            # model = GCN().to(DEVICE)
            # opt = torch.optim.Adam(model.parameters(), lr=args.lr)
            # criterion = nn.CrossEntropyLoss()
            if rounds == 0:
                tot_epoch = 5
            else:
                tot_epoch = args.epochs + 1
            for epoch in range(1, tot_epoch):
                model.train()

                correct = 0.
                total = 0.
                cnt = 0.
                total_loss = 0.
                total_expl = 0.
                total_ce = 0.
                average_aucs = 0.
                cnttt = 0.
                cnt_b = -1

                for batch in train_loader:
                    cnt_b += 1
                    batch = batch.to(DEVICE)
                    batch_size = batch.y.view(-1).size(0)
                    out = model(batch.x, batch.edge_index, batch.batch)

                    #plot all instances
                    # _, sal_b, _ = saliency_grad_diff(model, batch, epoch)
                    # node_imp_b = sal_b.sum(dim=1)

                    # data_list = batch.to_data_list()

                    # Plot each graph using ptr to slice node_imp correctly
                    # if epoch > 2:
                    #     for i, g in enumerate(data_list):
                    #         if i<4 and cnt_b<2:
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

                    if args.explainer == "ante":
                        expl_attn_logit, out = out # separate explanation from target predictions

                    correct += (out.argmax(dim=-1) == batch.y.view(-1)).sum().item()
                    total += batch.y.view(-1).size(0)
                    ce_loss = criterion(out, batch.y.view(-1))

                    graph_ids = batch.idx
                    mask = torch.tensor(
                        [gid.item() in explained_idx for gid in graph_ids],
                        device=batch.x.device
                    )

                    if mask.any():
                        node_mask = mask[batch.batch].bool()
                        sub_batch = Batch.from_data_list(batch.index_select(mask.nonzero(as_tuple=True)[0]))
                        gt_mask = batch.motif_node_mask[node_mask].to(DEVICE).float() if args.dataset == 'synth' else batch.node_label.bool()[node_mask].to(DEVICE).float()

                        if args.explainer == "post":
                            opt.zero_grad()
                            _, sal, aucs = saliency_grad_diff(model, sub_batch, epoch)

                            node_imp = sal.sum(dim=1)
                            reg = (node_imp ** 2).mean()
                            # positive mask: want high saliency
                            pos_loss = -torch.mean(node_imp[gt_mask.bool()])
                            # Negative mask: want low saliency
                            neg_loss = torch.mean(node_imp[~gt_mask.bool()])

                            # if rounds > 4 and epoch % 10 == 0 and cnttt == 0:
                            #     print('positives average: ',torch.mean(node_imp[gt_mask.bool()][0:6]))
                            #     print('confounder average:', node_imp[sub_batch.conf_id][0])
                            #     print('negatives average: ',torch.mean(node_imp[0:sub_batch.conf_id[0]]))
                            #     print(pos_loss, neg_loss)
                            #     graphs_in_sub_batch = sub_batch.to_data_list()
                            #     g0 = graphs_in_sub_batch[0]
                            #     plot_node_importance(g0, g0.motif_node_ids, g0.conf_id, node_imp[0:len(g0.y_color)],
                            #                        title="Node Importance")
                            #     # plot_g_tree(g0, trees, CID, node_imp[0:len(g0.y_color)])
                            # #     print(f'max node_imp {node_imp[0:len(g0.y_color)].max()},  and total average {node_imp[0:len(g0.y_color)].mean()}')

                            if args.only_neg:
                                expl_loss_sup = neg_loss
                            else:
                                expl_loss_sup = neg_loss + pos_loss

                            expl_loss = expl_loss_sup * mask.sum()/batch_size
                            cnttt += 1

                            log_dict = {
                                'batch_expl_loss': expl_loss,
                                'p_loss': pos_loss,
                                'n_loss': neg_loss
                            }
                            wandb.log(log_dict)

                        else:
                            expl_loss = F.binary_cross_entropy_with_logits(expl_attn_logit, gt_mask)

                            aucs = compute_plausibility(expl_attn_logit, sub_batch)
                        if aucs is not None:
                            average_aucs += aucs

                    else:
                        expl_loss = torch.tensor(0.0, device=DEVICE)
                        reg = torch.tensor(0.0, device=DEVICE)

                    loss = args.lam_ce * ce_loss + args.lam_expl * expl_loss #+ 0.01 * reg

                    opt.zero_grad()
                    loss.backward()
                    opt.step()

                    total_loss += float(loss.detach())
                    total_expl += float(expl_loss.detach())
                    total_ce += float(ce_loss.detach())

                tr_acc = correct / max(total, 1)
                total_loss = total_loss / max(len(train_loader), 1)
                total_expl = total_expl / max(len(train_loader), 1)
                total_ce = total_ce / max(len(train_loader), 1)
                average_aucs = average_aucs / max(len(train_loader), 1)

                model.eval()
                val_loss, val_acc = run_epoch(model, val_loader, opt, criterion, epoch, train=False, device=DEVICE)
                test_loss, test_acc = run_epoch(model, test_loader, opt, criterion, epoch, train=False, device=DEVICE)
                val_batch = Batch.from_data_list(val_set).to(DEVICE)

                if epoch % 5 == 0:
                    if args.explainer == "post":
                        _, _, val_aucs = saliency_grad_diff(model, val_batch, epoch)
                    else:
                        expl_attn_logit, out = model(val_batch.x, val_batch.edge_index, val_batch.batch)
                        val_aucs = compute_plausibility(expl_attn_logit, val_batch)

                    log_dict = {'epoch': epoch,
                                'total_loss_tr': total_loss,
                                'expl_loss': total_expl,
                                'ce_loss': total_ce,
                                'loss_val': val_loss,
                                'acc_tr': tr_acc,
                                'acc_val': val_acc,
                                'train_auc': average_aucs,
                                'val_auc': val_aucs}
                    wandb.log(log_dict)

                    print(f"Round {rounds} Epoch {epoch:02d} | "
                          f"train loss {total_loss:.3f}  ce loss {total_ce:.5f} expl loss {total_expl:.5f} reg {reg:.5f} "
                          f"acc {tr_acc:.3f} | val loss "
                          f"{val_loss:.3f} val acc {val_acc:.3f} test acc {test_acc}")
                    if val_aucs is None:
                        val_aucs = 0
                    print(f"train AUC {average_aucs:.3f} | val AUC {val_aucs: .3f}")

            # Chose graphs
            pool = list(all_idx - explained_idx)
            pool_dataset = torch.utils.data.Subset(train_set, pool)
            pool_loader = DataLoader(pool_dataset, batch_size=16, shuffle=False)

            # scores_m = uncertainty_scores_logits(model, pool_loader, DEVICE, method=args.active)
            scores_e = uncertainty_scores_e(model, pool_loader, DEVICE, method=args.active)
            chosen_id = select_topk(pool, scores_e, k=min(per_round, len(scores_e)))
            # conf_len = []
            # for j in chosen_id:
            #     conf_len.append(len(train_set[j].conf_id))
            # print(f'conf len average {sum(conf_len)/len(conf_len)}')
            # wandb.log({'conf len': sum(conf_len)/len(conf_len)})
            explained_idx.update(chosen_id)

            cpu_rng = torch.get_rng_state()
            cuda_rng = torch.cuda.get_rng_state() if torch.cuda.is_available() else None

            # # # plot the chosen graphs
            chosen_dataset = torch.utils.data.Subset(train_set, chosen_id)
            chosen_loader = DataLoader(chosen_dataset, batch_size=len(chosen_id), shuffle=False, num_workers=0)
            batch_c = next(iter(chosen_loader))

            _, sal_c, _ = saliency_grad_diff(model, batch_c, epoch=None)
            node_imp_c = sal_c.sum(dim=1)

            torch.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state(cuda_rng)

            # for graph_idx in range(batch_c.num_graphs):
            #     mask_c = (batch_c.batch == graph_idx)
            #     node_imp_g = node_imp_c[mask_c]
            #
            #     g = chosen_dataset[graph_idx]
            #
            #     plot_node_importance(
            #         g,
            #         g.motif_node_ids,
            #         g.conf_id,
            #         node_imp_g[: g.num_nodes],
            #         title="Node Importance of chosen",
            #     )

        total_val_acc = val_acc
        test_loss, test_acc = run_epoch(model, test_loader, opt, criterion, epoch, train=False, device=DEVICE)
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
