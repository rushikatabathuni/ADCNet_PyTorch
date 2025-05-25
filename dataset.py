import re
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from utils import smiles2adjoin, molecular_fg
from rdkit import Chem
from random import Random
from collections import defaultdict
from rdkit.Chem.Scaffolds import MurckoScaffold
from itertools import compress

str2num = {'<pad>':0 ,'H': 1, 'C': 2, 'N': 3, 'O': 4, 'S': 5, 'F': 6, 'Cl': 7, 'Br': 8, 'P':  9,
         'I': 10,'Na': 11,'B':12,'Se':13,'Si':14,'<unk>':15,'<mask>':16,'<global>':17}
num2str =  {i:j for j,i in str2num.items()}

def collate_fn_classification(batch):
    """Custom collate function for graph classification dataset"""
    x1_list, adj1_list, y_list, x2_list, adj2_list, idx_list = zip(*batch)
    
    # Pad sequences to same length within batch
    max_len1 = max([len(x) for x in x1_list])
    max_len2 = max([len(x) for x in x2_list])
    max_len = max(max_len1, max_len2)
    
    # Pad x sequences
    x1_padded = []
    x2_padded = []
    adj1_padded = []
    adj2_padded = []
    
    for i in range(len(batch)):
        # Pad x1
        x1_pad = torch.zeros(max_len, dtype=torch.long)
        x1_pad[:len(x1_list[i])] = x1_list[i]
        x1_padded.append(x1_pad)
        
        # Pad x2
        x2_pad = torch.zeros(max_len, dtype=torch.long)
        x2_pad[:len(x2_list[i])] = x2_list[i]
        x2_padded.append(x2_pad)
        
        # Pad adjacency matrices
        adj1_pad = torch.full((max_len, max_len), -1e9, dtype=torch.float32)
        adj1_pad[:adj1_list[i].shape[0], :adj1_list[i].shape[1]] = adj1_list[i]
        adj1_padded.append(adj1_pad)
        
        adj2_pad = torch.full((max_len, max_len), -1e9, dtype=torch.float32)
        adj2_pad[:adj2_list[i].shape[0], :adj2_list[i].shape[1]] = adj2_list[i]
        adj2_padded.append(adj2_pad)
    
    return (torch.stack(x1_padded),
            torch.stack(adj1_padded),
            torch.stack(y_list),
            torch.stack(x2_padded),
            torch.stack(adj2_padded),
            torch.stack(idx_list))

def collate_fn_inference(batch):
    """Custom collate function for inference dataset"""
    x_list, adj_list, smiles_list, atom_list = zip(*batch)
    
    # Pad sequences to same length within batch
    max_len = max([len(x) for x in x_list])
    
    x_padded = []
    adj_padded = []
    
    for i in range(len(batch)):
        # Pad x
        x_pad = torch.zeros(max_len, dtype=torch.long)
        x_pad[:len(x_list[i])] = x_list[i]
        x_padded.append(x_pad)
        
        # Pad adjacency matrix
        adj_pad = torch.full((max_len, max_len), -1e9, dtype=torch.float32)
        adj_pad[:adj_list[i].shape[0], :adj_list[i].shape[1]] = adj_list[i]
        adj_padded.append(adj_pad)
    
    return (torch.stack(x_padded),
            torch.stack(adj_padded),
            list(smiles_list),
            list(atom_list))

class GraphClassificationDataset(Dataset):
    """PyTorch Dataset for graph classification task"""
    def __init__(self, df, smiles_field1, smiles_field2, label_field, index_field, addH=True):
        self.df = df
        self.smiles_field1 = smiles_field1
        self.smiles_field2 = smiles_field2
        self.label_field = label_field
        self.index_field = index_field
        self.addH = addH
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        smiles1 = row[self.smiles_field1]
        smiles2 = row[self.smiles_field2]
        label = row[self.label_field]
        index = row[self.index_field]
        
        # Process first SMILES
        x1, adj1, y = self.numerical_smiles(smiles1, label)
        
        # Process second SMILES
        x2, adj2, idx = self.numerical_smiles(smiles2, index)
        
        return x1, adj1, y, x2, adj2, idx
    
    def numerical_smiles(self, smiles, label):
        atoms_list, adjoin_matrix = smiles2adjoin(smiles, explicit_hydrogens=self.addH)
        atoms_list = ['<global>'] + atoms_list
        nums_list = [str2num.get(i, str2num['<unk>']) for i in atoms_list]
        temp = np.ones((len(nums_list), len(nums_list)))
        temp[1:, 1:] = adjoin_matrix
        adjoin_matrix = (1 - temp) * (-1e9)
        x = torch.tensor(nums_list, dtype=torch.long)
        y = torch.tensor([label], dtype=torch.long)
        adjoin_matrix = torch.tensor(adjoin_matrix, dtype=torch.float32)
        return x, adjoin_matrix, y

class InferenceDataset(Dataset):
    """PyTorch Dataset for inference"""
    def __init__(self, sml_list, addH=True):
        self.sml_list = sml_list
        self.addH = addH
        
    def __len__(self):
        return len(self.sml_list)
    
    def __getitem__(self, idx):
        smiles = self.sml_list[idx]
        x, adj, smiles_out, atom_list = self.numerical_smiles(smiles)
        return x, adj, smiles_out, atom_list
    
    def numerical_smiles(self, smiles):
        smiles_origin = smiles
        atoms_list, adjoin_matrix = smiles2adjoin(smiles, explicit_hydrogens=self.addH)
        atoms_list = ['<global>'] + atoms_list
        nums_list = [str2num.get(i, str2num['<unk>']) for i in atoms_list]
        temp = np.ones((len(nums_list), len(nums_list)))
        temp[1:, 1:] = adjoin_matrix
        adjoin_matrix = (1 - temp) * (-1e9)
        x = torch.tensor(nums_list, dtype=torch.long)
        adjoin_matrix = torch.tensor(adjoin_matrix, dtype=torch.float32)
        return x, adjoin_matrix, smiles, atoms_list

class Graph_Classification_Dataset(object):  # Graph classification task data set processing
    def __init__(self, path, smiles_field1='Smiles1', smiles_field2='Smiles2', label_field='label', 
                 index_field='label', max_len=500, seed=1, batch_size=16, a=1, addH=True):
        if path.endswith('.txt') or path.endswith('.tsv'):
            self.df = pd.read_csv(path, sep='\t', encoding='latin1')
        elif path.endswith('.xlsx'):
            self.df = pd.read_excel(path, engine="openpyxl")
        else:
            self.df = pd.read_csv(path, encoding='latin1')
        
        self.smiles_field1 = smiles_field1
        self.smiles_field2 = smiles_field2
        self.label_field = label_field
        self.index_field = index_field
        self.vocab = str2num
        self.devocab = num2str
        self.df = self.df[self.df[smiles_field1].str.len() <= max_len]
        self.df = self.df[[True if Chem.MolFromSmiles(smi) is not None else False for smi in self.df[smiles_field1]]]
        self.seed = seed
        self.batch_size = batch_size
        self.a = a
        self.addH = addH
        
        # Set random seed for reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)

    def get_data(self):
        '''Randomized Split Dataset'''
        data = self.df
        data = data.fillna(666)
        
        # Split data
        train_idx = []
        idx = data.sample(frac=0.8, random_state=self.seed).index
        train_idx.extend(idx)
        train_data = data[data.index.isin(train_idx)]
        data = data[~data.index.isin(train_idx)]
        
        test_idx = []
        idx = data[~data.index.isin(train_data)].sample(frac=0.5, random_state=self.seed).index
        test_idx.extend(idx)
        test_data = data[data.index.isin(test_idx)]
        val_data = data[~data.index.isin(train_idx + test_idx)]
        
        df_train_data = pd.DataFrame(train_data).reset_index(drop=True)
        df_test_data = pd.DataFrame(test_data).reset_index(drop=True)
        df_val_data = pd.DataFrame(val_data).reset_index(drop=True)

        # Create PyTorch datasets
        train_dataset = GraphClassificationDataset(
            df_train_data, self.smiles_field1, self.smiles_field2, 
            self.label_field, self.index_field, self.addH
        )
        test_dataset = GraphClassificationDataset(
            df_test_data, self.smiles_field1, self.smiles_field2, 
            self.label_field, self.index_field, self.addH
        )
        val_dataset = GraphClassificationDataset(
            df_val_data, self.smiles_field1, self.smiles_field2, 
            self.label_field, self.index_field, self.addH
        )

        # Create DataLoaders
        self.dataset1 = DataLoader(
            train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            collate_fn=collate_fn_classification,
            num_workers=0,  # Set to 0 to avoid multiprocessing issues
            pin_memory=True
        )
        
        self.dataset2 = DataLoader(
            test_dataset, 
            batch_size=512, 
            shuffle=False, 
            collate_fn=collate_fn_classification,
            num_workers=0,
            pin_memory=True
        )
        
        self.dataset3 = DataLoader(
            val_dataset, 
            batch_size=512, 
            shuffle=False, 
            collate_fn=collate_fn_classification,
            num_workers=0,
            pin_memory=True
        )

        return self.dataset1, self.dataset2, self.dataset3

class Inference_Dataset(object):
    def __init__(self, sml_list, max_len=500, addH=True):
        self.vocab = str2num
        self.devocab = num2str
        self.sml_list = [i for i in sml_list if len(i) < max_len]
        self.addH = addH

    def get_data(self):
        # Create PyTorch dataset
        dataset = InferenceDataset(self.sml_list, self.addH)
        
        # Create DataLoader
        self.dataset = DataLoader(
            dataset, 
            batch_size=512, 
            shuffle=False, 
            collate_fn=collate_fn_inference,
            num_workers=0,
            pin_memory=True
        )

        return self.dataset
