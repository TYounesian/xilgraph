import os
import os.path as osp
import pickle
import gdown

import numpy as np
import torch
from torch_geometric.data import InMemoryDataset, Data, extract_zip
from torch_geometric.utils import dense_to_sparse

from scipy.spatial.distance import cdist



class CPatchMNIST(InMemoryDataset):
    r"""
        Adding colored patches to the MNIST75sp dataset
    """

    def __init__(self, root: str, mode: str = 'train', transform=None, pre_transform=None):

        self.name = self.__class__.__name__

        self.node_gt_att_threshold = 0
        self.use_mean_px = True
        self.use_coord = True
        self.mode = mode

        self.color_mapping = {
            0: np.array([255,  0,  0]), # Red
            1: np.array([255,128,  0]), # Orange
            2: np.array([255,255,  0]), # Yellow
            3: np.array([  0,255,  0]), # Green
            4: np.array([ 51,255,255]), # Light Blue
            5: np.array([127,  0,255]), # Violet
            6: np.array([  0,  0,  0]), # Black
            7: np.array([255,255,255]), # White
            8: np.array([255,204,255]), # Pink
            9: np.array([  0,  0,255]), # Blue
        }
        self.color_mapping = {k: v / 255 for k, v in self.color_mapping.items()} # normalize values

        super(CPatchMNIST, self).__init__(root, transform, pre_transform, None)

        idx = self.processed_file_names.index('cpatchmnist_75sp_{}.pt'.format(self.mode))
        self.data, self.slices = torch.load(self.processed_paths[idx], weights_only=False)

    @property
    def raw_file_names(self):
        return ['mnist_75sp_train.pkl', 'mnist_75sp_test.pkl']

    @property
    def processed_file_names(self):
        return ['cpatchmnist_75sp_train.pt', 'cpatchmnist_75sp_test.pt', 'cpatchmnist_75sp_id_test.pt']

    def download(self):
        for file in self.raw_file_names:
            if not osp.exists(osp.join(self.raw_dir, file)):
                print("raw data of `{}` doesn't exist, please download from our github.".format(file))
                raise FileNotFoundError

    def process(self):
        if self.mode[:3] == "id_":
            suffix = self.mode[3:]
        else:
            suffix = self.mode

        data_file = 'mnist_75sp_%s.pkl' % suffix
        with open(osp.join(self.raw_dir, data_file), 'rb') as f:
            self.labels, self.sp_data = pickle.load(f)

        sp_file = 'mnist_75sp_%s_superpixels.pkl' % suffix
        with open(osp.join(self.raw_dir, sp_file), 'rb') as f:
            self.all_superpixels = pickle.load(f)

        self.use_mean_px = self.use_mean_px
        self.use_coord = self.use_coord
        self.n_samples = len(self.labels)
        self.img_size = 28

        self.edge_indices, self.xs, self.edge_attrs, self.node_gt_atts, self.edge_gt_atts = [], [], [], [], []
        data_list = []
        for index, sample in enumerate(self.sp_data):
            mean_px, coord, sp_order = sample[:3]
            superpixels = self.all_superpixels[index]
            coord = coord / self.img_size
            A = self.compute_adjacency_matrix_images(coord)
            N_nodes = A.shape[0]

            A = torch.FloatTensor((A > 0.1) * A)
            edge_index, edge_attr = dense_to_sparse(A)

            x = None
            if self.use_mean_px:
                x = mean_px.reshape(N_nodes, -1)           

            if self.use_coord:
                coord = coord.reshape(N_nodes, 2)
                if self.use_mean_px:
                    x = np.concatenate((x, coord), axis=1)
                else:
                    x = coord
                    
            if x is None:
                assert False
                x = np.ones(N_nodes, 1)  # dummy features

            # replicate features to make it possible to test on colored images
            x = np.pad(x, ((0, 0), (2, 0)), 'edge')
            if self.node_gt_att_threshold == 0:
                node_gt_att = (mean_px > 0).astype(np.float32)
            else:
                node_gt_att = mean_px.copy()
                node_gt_att[node_gt_att < self.node_gt_att_threshold] = 0

            node_gt_att = torch.LongTensor(node_gt_att).view(-1)
            row, col = edge_index
            edge_gt_att = torch.LongTensor(node_gt_att[row] * node_gt_att[col]).view(-1)

            # Adding colored patch in first and last superpixel
            if self.mode == "test":
                x[sp_order == 0, :3] = self.color_mapping[(self.labels[index] + 1) % 9]
                x[sp_order == max(sp_order), :3] = self.color_mapping[(self.labels[index] + 1) % 9]
            elif self.mode == "id_test":
                x[node_gt_att.bool(), :3] = 0.0
            else:
                x[sp_order == 0, :3] = self.color_mapping[self.labels[index]]
                x[sp_order == max(sp_order), :3] = self.color_mapping[self.labels[index]]

            data_list.append(
                Data(
                    x=torch.tensor(x),
                    y=torch.LongTensor([self.labels[index]]),
                    edge_index=edge_index,
                    edge_attr=edge_attr.reshape(-1, 1),
                    node_label=node_gt_att.float(),
                    edge_label=edge_gt_att.float(),
                    sp_order=torch.tensor(sp_order),
                    superpixels=torch.tensor(superpixels),
                    name=f'CPatchMNISTSP-{self.mode}-{index}', idx=index
                )
            )
        idx = self.processed_file_names.index('cpatchmnist_75sp_{}.pt'.format(self.mode))

        torch.save(self.collate(data_list), self.processed_paths[idx])

    def compute_adjacency_matrix_images(self, coord, sigma=0.1):
        coord = coord.reshape(-1, 2)
        dist = cdist(coord, coord)
        A = np.exp(- dist / (sigma * np.pi) ** 2)
        A[np.diag_indices_from(A)] = 0
        return A

    @staticmethod
    def load(dataset_root: str):
        train_set = CPatchMNIST(dataset_root + "/CPatchMNIST/", mode="train") # MNIST with color patch
        test_set = CPatchMNIST(dataset_root + "/CPatchMNIST/", mode="test")  # MNIST with color patch
        id_test_set = CPatchMNIST(dataset_root + "/CPatchMNIST/", mode="id_test")  # MNIST without color patch

        n_train_data, n_val_data = 20000, 5000
        perm_idx = torch.randperm(len(train_set))     
        train_val = train_set[perm_idx]   

        train_dataset = train_val[:n_train_data]
        id_val_dataset = train_val[-n_val_data:]
        id_test_dataset = id_test_set
        test_dataset = test_set

        return {'train': train_dataset, 'id_val': id_val_dataset, 'id_test': id_test_dataset,
                'val': id_val_dataset, 'test': test_dataset}

class GOODMotif(InMemoryDataset):
    r"""
    The GOOD-Motif dataset motivated by `Spurious-Motif
    <https://arxiv.org/abs/2201.12872>`_.

    Args:
        root (str): The dataset saving root.
        domain (str): The domain selection. Allowed: 'basis' and 'size'.
        shift (str): The distributional shift we pick. Allowed: 'no_shift', 'covariate', and 'concept'.
        subset (str): The split set. Allowed: 'train', 'id_val', 'id_test', 'val', and 'test'. When shift='no_shift',
            'id_val' and 'id_test' are not applicable.
        generate (bool): The flag for regenerating dataset. True: regenerate. False: download.
    """

    def __init__(self, root: str, domain: str, shift: str = 'no_shift', subset: str = 'train', transform=None,
                 pre_transform=None, generate: bool = False, debias=False):

        self.name = self.__class__.__name__
        self.domain = domain
        self.url = 'https://drive.google.com/file/d/15YRuZG6wI4HF7QgrLI52POKjuObsOyvb/view?usp=sharing'

        self.generate = generate

        self.all_basis = ["wheel", "tree", "ladder", "star", "path"]
        self.basis_role_end = {'wheel': 0, 'tree': 0, 'ladder': 0, 'star': 1, 'path': 1}
        self.all_motifs = [[["house"]], [["dircycle"]], [["crane"]]]
        self.num_data = 30000
        self.train_spurious_ratio = [0.99, 0.97, 0.95]

        super().__init__(root, transform, pre_transform)

        if shift == 'covariate':
            subset_pt = 3
        elif shift == 'concept':
            subset_pt = 8
        elif shift == 'no_shift':
            subset_pt = 0
        else:
            raise ValueError(f'Unknown shift: {shift}.')
        if subset == 'train':
            subset_pt += 0
        elif subset == 'val':
            subset_pt += 1
        elif subset == 'test':
            subset_pt += 2
        elif subset == 'id_val':
            subset_pt += 3
        else:
            subset_pt += 4

        self.data, self.slices = torch.load(self.processed_paths[subset_pt], weights_only=False)

    @property
    def raw_dir(self):
        return osp.join(self.root)

    def _download(self):
        if os.path.exists(osp.join(self.raw_dir, self.name)) or self.generate:
            return
        if not os.path.exists(self.raw_dir):
            os.makedirs(self.raw_dir)
        self.download()

    def download(self):
        path = gdown.download(self.url, output=osp.join(self.raw_dir, self.name + '.zip'), fuzzy=True)
        extract_zip(path, self.raw_dir)
        os.unlink(path)

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, self.domain, 'processed')

    @property
    def processed_file_names(self):
        return ['no_shift_train.pt', 'no_shift_val.pt', 'no_shift_test.pt',
                'covariate_train.pt', 'covariate_val.pt', 'covariate_test.pt', 'covariate_id_val.pt', 'covariate_id_test.pt',
                'concept_train.pt', 'concept_val.pt', 'concept_test.pt', 'concept_id_val.pt', 'concept_id_test.pt']

    @staticmethod
    def load(dataset_root: str):
        domain = "basis"
        shift = "covariate" # I think you can keep this value

        train_dataset = GOODMotif(root=dataset_root, domain=domain, shift=shift, subset='train') # ID train
        id_val_dataset = GOODMotif(root=dataset_root, domain=domain, shift=shift, subset='id_val') # ID val
        id_test_dataset = GOODMotif(root=dataset_root, domain=domain, shift=shift, subset='id_test') # ID test
        val_dataset = GOODMotif(root=dataset_root, domain=domain, shift=shift, subset='val') # OOD val
        test_dataset = GOODMotif(root=dataset_root, domain=domain, shift=shift, subset='test') # OOD test

        return {'train': train_dataset, 'id_val': id_val_dataset, 'id_test': id_test_dataset,
                'val': val_dataset, 'test': test_dataset}


if __name__ == "__main__":
    train_dataset = CPatchMNIST.load(dataset_root="./data")["train"]
    print(train_dataset)

    train_dataset = GOODMotif.load(dataset_root="./data")["train"]
    print(train_dataset)