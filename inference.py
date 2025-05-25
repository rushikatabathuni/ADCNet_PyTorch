import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataset import Graph_Classification_Dataset, Inference_Dataset
from sklearn.metrics import roc_auc_score, confusion_matrix, precision_recall_curve, auc
from rdkit.Chem import Draw
import os
import pickle
from sklearn.preprocessing import StandardScaler
import math

def cover_dict(path):
    file_path = path
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    tensor_dict = {key: torch.tensor(value, dtype=torch.float32) for key, value in data.items()}
    new_data = {i: value for i, (key, value) in enumerate(tensor_dict.items())}
    return new_data

def score(y_test, y_pred):
    auc_roc_score = roc_auc_score(y_test, y_pred)
    prec, recall, _ = precision_recall_curve(y_test, y_pred)
    prauc = auc(recall, prec)
    y_pred_print = [round(y, 0) for y in y_pred]
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_print).ravel()
    se = tp / (tp + fn)
    sp = tn / (tn + fp) 
    acc = (tp + tn) / (tp + fn + tn + fp)
    mcc = (tp * tn - fn * fp) / math.sqrt((tp + fn) * (tp + fp) * (tn + fn) * (tn + fp))
    P = tp / (tp + fp)
    F1 = (P * se * 2) / (P + se)
    BA = (se + sp) / 2
    PPV = tp / (tp + fp)
    NPV = tn / (fn + tn)
    return tp, tn, fn, fp, se, sp, mcc, acc, auc_roc_score, F1, BA, prauc, PPV, NPV

def DAR_feature(file_path, column_name):
    df = pd.read_excel(file_path)
    column_data = df[column_name].values.reshape(-1, 1)
    mean_value = 3.86845977
    variance_value = 1.569108443
    std_deviation = variance_value**0.5
    column_data_standardized = (column_data - mean_value) / std_deviation
    normalized_data = (column_data_standardized - 0.8) / (12 - 0.8)
    data_dict = {index: torch.tensor(value, dtype=torch.float32) for index, value in zip(df.index, normalized_data.flatten())}
    return data_dict

# Load feature dictionaries
Heavy_dict = cover_dict('Heavy.pkl')
Light_dict = cover_dict('Light.pkl')
Antigen_dict = cover_dict('Antigen.pkl')
DAR_dict = DAR_feature('data.xlsx', 'DAR')

# Model configuration
medium = {'name':'Medium','num_layers': 6, 'num_heads': 8, 'd_model': 256,'path':'medium_weights','addH':True}
arch = medium
trained_epoch = 20
num_layers = arch['num_layers']
num_heads = arch['num_heads']
d_model = arch['d_model']
addH = arch['addH']
dff = d_model * 2
vocab_size = 18
dense_dropout = 0.1
seed = 1

np.random.seed(seed=seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

# Load data
df = pd.read_excel('data.xlsx')
sml_list1 = df['Payload Isosmiles'].tolist()
sml_list2 = df['Linker Isosmiles'].tolist()
ans = []
y_preds = []
res = []
n = len(sml_list1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from model import PredictModel
model = PredictModel(num_layers=num_layers,
                     d_model=d_model,
                     dff=dff, 
                     num_heads=num_heads, 
                     vocab_size=vocab_size,
                     a=1,
                     dense_dropout=dense_dropout)


model.load_state_dict(torch.load('classification_weights/ADC_9.pth', map_location=device))
model.to(device)
model.eval()

with torch.no_grad():
    for i in range(n):
        x1 = [sml_list1[i]]
        x2 = [sml_list2[i]]
        t1 = Heavy_dict[i]
        t2 = Light_dict[i]
        t3 = Antigen_dict[i]
        t4 = DAR_dict[i].item()

        t1 = t1.unsqueeze(0)
        t2 = t2.unsqueeze(0)
        t3 = t3.unsqueeze(0)
        t4 = torch.tensor([[t4]], dtype=torch.float32)  # shape (1, 1)

        inference_dataset1 = Inference_Dataset(x1, addH=addH).get_data()
        inference_dataset2 = Inference_Dataset(x2, addH=addH).get_data()

        x1_batch, adjoin_matrix1_batch, smiles1, atom_list1 = next(iter(inference_dataset1))
        x2_batch, adjoin_matrix2_batch, smiles2, atom_list2 = next(iter(inference_dataset2))
        
        x1_tensor = x1_batch[0:1]
        adjoin_matrix1 = adjoin_matrix1_batch[0:1]
        x2_tensor = x2_batch[0:1]
        adjoin_matrix2 = adjoin_matrix2_batch[0:1]

        seq1 = (x1_tensor == 0).float()
        seq2 = (x2_tensor == 0).float()

        mask1 = seq1.unsqueeze(1).unsqueeze(1)
        mask2 = seq2.unsqueeze(1).unsqueeze(1)

        x1_tensor = x1_tensor.to(device)
        x2_tensor = x2_tensor.to(device)
        adjoin_matrix1 = adjoin_matrix1.to(device)
        adjoin_matrix2 = adjoin_matrix2.to(device)
        mask1 = mask1.to(device)
        mask2 = mask2.to(device)
        t1 = t1.to(device)
        t2 = t2.to(device)
        t3 = t3.to(device)
        t4 = t4.to(device)

        # Model inference
        x = model(x1=x1_tensor, 
                  mask1=mask1, 
                  training=False,
                  adjoin_matrix1=adjoin_matrix1, 
                  x2=x2_tensor, 
                  mask2=mask2, 
                  adjoin_matrix2=adjoin_matrix2,
                  t1=t1,
                  t2=t2,
                  t3=t3,
                  t4=t4)
        
        y_preds.append(x)

y_preds = torch.cat(y_preds, dim=0)
y_preds = torch.sigmoid(y_preds)
y_preds = y_preds.flatten()

y_hat = torch.where(y_preds < 0.5, 
                    torch.tensor(0, dtype=torch.long), 
                    torch.tensor(1, dtype=torch.long))

for i in y_preds.cpu().numpy():
    ans.append(i)
for i in y_hat.cpu().numpy():
    res.append(i)

print(ans)
print(res)