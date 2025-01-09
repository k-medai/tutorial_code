import os
import os.path as osp

import re
import torch
import pandas as pd
from rdkit import Chem
from typing import Callable, Dict, Optional, Tuple, Union, List
from torch_geometric.data import InMemoryDataset, download_url, extract_gz, Data

import warnings
warnings.filterwarnings('ignore')

# Atom feature sizes (From KANO/chempromp/feature/featurization.py)
MAX_ATOMIC_NUM = 100
ATOM_FEATURES = {
    'atomic_num': list(range(MAX_ATOMIC_NUM)),
    'degree': [0, 1, 2, 3, 4, 5],
    'formal_charge': [-1, -2, 1, 2, 0],
    'chiral_tag': [0, 1, 2, 3],
    'num_Hs': [0, 1, 2, 3, 4],
    'hybridization': [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2
    ],
}

# from chem/loader.py
EDGE_FEATURES = {
    'possible_bonds': [
        Chem.rdchem.BondType.UNSPECIFIED,
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.QUADRUPLE,
        Chem.rdchem.BondType.QUINTUPLE,
        Chem.rdchem.BondType.HEXTUPLE,
        Chem.rdchem.BondType.ONEANDAHALF,
        Chem.rdchem.BondType.TWOANDAHALF,
        Chem.rdchem.BondType.THREEANDAHALF,
        Chem.rdchem.BondType.FOURANDAHALF,
        Chem.rdchem.BondType.FIVEANDAHALF,
        Chem.rdchem.BondType.AROMATIC,
        Chem.rdchem.BondType.IONIC,
        Chem.rdchem.BondType.HYDROGEN,
        Chem.rdchem.BondType.THREECENTER,
        Chem.rdchem.BondType.DATIVEONE,
        Chem.rdchem.BondType.DATIVE,
        Chem.rdchem.BondType.DATIVEL,
        Chem.rdchem.BondType.DATIVER,
        Chem.rdchem.BondType.OTHER,
        Chem.rdchem.BondType.ZERO,
    ],
    'possible_bond_dirs': [  # only for double bond stereo information
        Chem.rdchem.BondDir.NONE,
        Chem.rdchem.BondDir.ENDUPRIGHT,
        Chem.rdchem.BondDir.ENDDOWNRIGHT
    ]
}


def feature_to_onehot(value: int, choices: List[int]) -> List[int]:
    r"""
    From KANO/chempromp/feature/featurization.py > onek_encoding_unk()
    Creates a one-hot encoding.

    value: The value for which the encoding should be one.
    choices: A list of possible values
    return: A one-hot encoding of the value in a laist of length len(choices) + 1
    If value is not in the list of choices, then the final emement in the encoding is 1
    """
    encoding = [0] * (len(choices) + 1)
    index = choices.index(value) if value in choices else -1
    encoding[index] = 1

    return encoding


def atom_features(atom: Chem.rdchem.Atom) -> List[Union[bool, int, float]]:
    r"""
    From KANO/chempromp/feature/featurization.py > atom_features()
    Builds a feature vector for an atom

    node feature
    : 원자 번호, degree, formalCharge, 카이랄성, 수소 수, Hybridization, 방향족 여부, 질량
    방향족 여부와 질량 제외 one-hot으로 입력
    """
    features = feature_to_onehot(atom.GetAtomicNum() - 1, ATOM_FEATURES['atomic_num']) + \
               feature_to_onehot(atom.GetTotalDegree(), ATOM_FEATURES['degree']) + \
               feature_to_onehot(atom.GetFormalCharge(), ATOM_FEATURES['formal_charge']) + \
               feature_to_onehot(int(atom.GetChiralTag()), ATOM_FEATURES['chiral_tag']) + \
               feature_to_onehot(int(atom.GetTotalNumHs()), ATOM_FEATURES['num_Hs']) + \
               feature_to_onehot(int(atom.GetHybridization()), ATOM_FEATURES['hybridization']) + \
               [1 if atom.GetIsAromatic() else 0] + \
               [atom.GetMass() * 0.01]  # scaled to about the same range as other features
    return features


def smiles_to_feature(smiles: str, with_hydrogen: bool = False,
                      kekulize: bool = False) -> 'torch_geometric.data.Data':
    r"""Converts a SMILES string to a :class:`torch_geometric.data.Data`
    instance.

    Args:
        smiles (str): The SMILES string.
        with_hydrogen (bool, optional): If set to :obj:`True`, will store
            hydrogens in the molecule graph. (default: :obj:`False`)
        kekulize (bool, optional): If set to :obj:`True`, converts aromatic
            bonds to single/double bonds. (default: :obj:`False`)
    """
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        mol = Chem.MolFromSmiles('')
    if with_hydrogen:
        mol = Chem.AddHs(mol)
    if kekulize:
        Chem.Kekulize(mol)

    xs: List[List[int]] = []
    for atom in mol.GetAtoms():
        current_atom_feat = atom_features(atom)
        xs.append(current_atom_feat)

    x = torch.tensor(xs, dtype=torch.long).view(-1, 133)

    edge_indices, edge_attrs = [], []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        edge_feature = [EDGE_FEATURES['possible_bonds'].index(bond.GetBondType())] + [
            EDGE_FEATURES['possible_bond_dirs'].index(bond.GetBondDir())]

        edge_indices += [[i, j], [j, i]]
        edge_attrs += [edge_feature, edge_feature]

    edge_index = torch.tensor(edge_indices)
    edge_index = edge_index.t().to(torch.long).view(2, -1)
    edge_attr = torch.tensor(edge_attrs, dtype=torch.long).view(-1, 2)

    if edge_index.numel() > 0:  # Sort indices.
        perm = (edge_index[0] * x.size(0) + edge_index[1]).argsort()
        edge_index, edge_attr = edge_index[:, perm], edge_attr[perm]

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, smiles=smiles)


class CustomMoleculeNet(InMemoryDataset):
    r"""The `MoleculeNet <http://moleculenet.org/datasets-1>`_ benchmark
    collection  from the `"MoleculeNet: A Benchmark for Molecular Machine
    Learning" <https://arxiv.org/abs/1703.00564>`_ paper, containing datasets
    from physical chemistry, biophysics and physiology.
    All datasets come with the additional node and edge features introduced by
    the :ogb:`null`
    `Open Graph Benchmark <https://ogb.stanford.edu/docs/graphprop/>`_.

    Args:
        root (str): Root directory where the dataset should be saved.
        name (str): The name of the dataset (:obj:`"ESOL"`, :obj:`"FreeSolv"`,
            :obj:`"Lipo"`, :obj:`"PCBA"`, :obj:`"MUV"`, :obj:`"HIV"`,
            :obj:`"BACE"`, :obj:`"BBBP"`, :obj:`"Tox21"`, :obj:`"ToxCast"`,
            :obj:`"SIDER"`, :obj:`"ClinTox"`).
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)
        force_reload (bool, optional): Whether to re-process the dataset.
            (default: :obj:`False`)
    """

    url = 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/{}'

    # Format: name: (display_name, url_name, csv_name, smiles_idx, y_idx)
    names: Dict[str, Tuple[str, str, str, int, Union[int, slice]]] = {
        'esol': ('ESOL', 'delaney-processed.csv', 'delaney-processed', -1, -2),
        'freesolv': ('FreeSolv', 'SAMPL.csv', 'SAMPL', 1, 2),
        'lipo': ('Lipophilicity', 'Lipophilicity.csv', 'Lipophilicity', 2, 1),
        'pcba': ('PCBA', 'pcba.csv.gz', 'pcba', -1, slice(0, 128)),
        'muv': ('MUV', 'muv.csv.gz', 'muv', -1, slice(0, 17)),
        'hiv': ('HIV', 'HIV.csv', 'HIV', 0, -1),
        'bace': ('BACE', 'bace.csv', 'bace', 0, 2),
        'bbbp': ('BBBP', 'BBBP.csv', 'BBBP', -1, -2),
        'tox21': ('Tox21', 'tox21.csv.gz', 'tox21', -1, slice(0, 12)),
        'toxcast':
            ('ToxCast', 'toxcast_data.csv.gz', 'toxcast_data', 0, slice(1, 618)),
        'sider': ('SIDER', 'sider.csv.gz', 'sider', 0, slice(1, 28)),
        'clintox': ('ClinTox', 'clintox.csv.gz', 'clintox', 0, slice(1, 3)),
    }

    def __init__(
            self,
            root: str,
            name: str,
            transform: Optional[Callable] = None,
            pre_transform: Optional[Callable] = None,
            pre_filter: Optional[Callable] = None,
            # force_reload: bool = False,
    ) -> None:
        self.name = name.lower()
        assert self.name in self.names.keys()
        super().__init__(root, transform, pre_transform, pre_filter)
        # self.load(self.processed_paths[0]) # 데이터 다운로드 및 전처리 후 바로 사용할 수 있도록 메모리에 로드하는 역할

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self) -> str:
        return f'{self.names[self.name][2]}.csv'

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self) -> None:
        url = self.url.format(self.names[self.name][1])
        path = download_url(url, self.raw_dir)
        if self.names[self.name][1][-2:] == 'gz':
            extract_gz(path, self.raw_dir)
            os.unlink(path)

    def process(self) -> None:
        with open(self.raw_paths[0], 'r') as f:
            dataset = f.read().split('\n')[1:-1]
            dataset = [x for x in dataset if len(x) > 0]  # Filter empty lines.

        smiles_list = [] # smiles 저장 추가
        labels_list = [] # label 저장 추가
        
        for line in dataset:
            line = re.sub(r'\".*\"', '', line)  # Replace ".*" strings.
            values = line.split(',')

            smiles = values[self.names[self.name][3]]
            labels = values[self.names[self.name][4]]
            labels = labels if isinstance(labels, list) else [labels]

            # remove salts
            smiles = max(smiles.split('.'), key=len)
            
            # remove molecule without label
            ys = [float(y) if len(y) > 0 else '-' for y in labels]
            if '-' in ys:
                print("{} is removed (no label)".format(data.smiles))
                continue

            y = torch.tensor(ys, dtype=torch.float).view(1, -1)
            
            smiles_list.append(smiles)
            labels_list.append(y)

        # check the save path 
        # save smiles and labels toghter

        mapped_data = [{"smiles": smi, "label": lbl.tolist()} for smi, lbl in zip(smiles_list, labels_list)]
        torch.save(mapped_data, os.path.join(self.processed_dir, 'smiles_labels.pt'))

        
        # smiles_path = os.path.join(self.processed_dir, 'smiles.csv')
        # labels_path = os.path.join(self.processed_dir, 'labels.pt')
    
        # # Ensure the directory exists
        # os.makedirs(self.processed_dir, exist_ok=True)
    
        # # save smiles and labels
        # pd.Series(smiles_list).to_csv(os.path.join(self.processed_dir, 'smiles.csv'), index=False, header=False)
        # torch.save(labels_list, os.path.join(self.processed_dir, 'labels.pt'))

        print("Data processed and saved successfully.")
        
        #     # generate features
        #     data = smiles_to_feature(smiles)

        #     # remove molecule with too simple atoms - ex: C, CC
        #     if data.x.shape[0] < 3:
        #         print("{} is removed (too simple atoms)".format(data.smiles))
        #         continue


        #     data.y = y

        #     if self.pre_filter is not None and not self.pre_filter(data):
        #         continue

        #     if self.pre_transform is not None:
        #         data = self.pre_transform(data)

        #     data_list.append(data)
        #     data_smiles_list.append(smiles)  ## save smiles list

        # ## save smiles list
        # data_smiles_series = pd.Series(data_smiles_list)
        # data_smiles_series.to_csv(os.path.join(self.processed_dir,
        #                                        'smiles.csv'), index=False, header=False)
        # self.save(data_list, self.processed_paths[0])
    
    def __repr__(self) -> str:
        return f'{self.names[self.name][0]}({len(self)})'