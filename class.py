import sklearn
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from dataset import Graph_Classification_Dataset
import os
import pandas as pd
from model import PredictModel
from sklearn.metrics import roc_auc_score, confusion_matrix, precision_recall_curve, auc
from hyperopt import fmin, tpe, hp
from utils import get_task_names
from sklearn.preprocessing import StandardScaler
import pickle
import math
import csv

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

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
    df = pd.read_excel(file_path, engine='openpyxl')
    column_data = df[column_name].values.reshape(-1, 1)
    scaler = StandardScaler()
    column_data_standardized = scaler.fit_transform(column_data)
    # PyTorch equivalent of tf.keras.utils.normalize
    column_data_normalized = F.normalize(torch.tensor(column_data_standardized, dtype=torch.float32), dim=0).flatten()
    data_dict = {index: column_data_normalized[index] for index in df.index}
    return data_dict

def process_list(input_list):
    input_list.append(np.mean(input_list))
    mean_value = np.mean(input_list[:-1])
    std_value = np.std(input_list[:-1], ddof=0)
    mean_range = f'{mean_value:.4f} ± {std_value:.4f}'
    input_list[-1] = mean_range
    print(input_list)
    return input_list

def extract_tensors(index, heavy_dict, light_dict, antigen_dict, dar_dict):
    heavy_tensor_list = []
    light_tensor_list = []
    antigen_tensor_list = []
    DAR_tensor_list = []

    if torch.is_tensor(index):
        index_np = index.cpu().numpy()
    else:
        index_np = index

    index_np = np.squeeze(index_np) 

    for idx in index_np:
        heavy_tensor_list.append(heavy_dict[idx])
        light_tensor_list.append(light_dict[idx])
        antigen_tensor_list.append(antigen_dict[idx])
        DAR_tensor_list.append(dar_dict[idx])
    
    t1 = torch.stack(heavy_tensor_list)
    t2 = torch.stack(light_tensor_list)
    t3 = torch.stack(antigen_tensor_list)
    t4 = torch.stack(DAR_tensor_list)
    
    return t1.to(device), t2.to(device), t3.to(device), t4.to(device)

Heavy_dict = cover_dict('Embeddings/Heavy_1280.pkl')
Light_dict = cover_dict('Embeddings/Light_1280.pkl')
Antigen_dict = cover_dict('Embeddings/Antigen_1280.pkl')
DAR_dict = DAR_feature('data.xlsx', 'DAR_val')

def main(seed, args):
    task = 'ADC'
    idx = ['index']
    label = ['label（100nm）']

    arch = {'name': 'Medium', 'path': 'medium3_weights'}
    trained_epoch = 20
    num_layers = 6
    d_model = 256
    addH = True
    dff = d_model * 2
    vocab_size = 18

    num_heads = args['num_heads']
    dense_dropout = args['dense_dropout']
    learning_rate = args['learning_rate']
    batch_size = args['batch_size']
    
    # Set random seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    train_dataset, test_dataset, val_dataset = Graph_Classification_Dataset('data.xlsx', 
                                                                            smiles_field1='Payload Isosmiles',
                                                                            smiles_field2='Linker Isosmiles',
                                                                            label_field=label,
                                                                            index_field=idx, 
                                                                            seed=seed,
                                                                            batch_size=batch_size,
                                                                            a=len(label), 
                                                                            addH=addH).get_data()

    sample_batch = next(iter(train_dataset))
    x1, adjoin_matrix1, y, x2, adjoin_matrix2, index = sample_batch

    model = PredictModel(num_layers=num_layers,
                         d_model=d_model,
                         dff=dff, 
                         num_heads=num_heads, 
                         vocab_size=vocab_size,
                         dropout_rate=dense_dropout)
    
    model = model.to(device)

    if os.path.exists("Weights/Encoder_Weights.pth"):
        encoder_state_dict = torch.load("Weights/Encoder_Weights.pth", 
                                      map_location=device)
        model.encoder.load_state_dict(encoder_state_dict)
        print('Loaded pretrained encoder weights')

    total_params = count_parameters(model)
    print('*' * 100)
    print("Total Parameters:", total_params)
    print('*' * 100)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()

    best_auc = -10
    stopping_monitor = 0
    
    for epoch in range(200):
        model.train()
        epoch_loss = 0
        num_batches = 0
        
        for x1, adjoin_matrix1, y, x2, adjoin_matrix2, index in train_dataset:
            # Move to device
            x1, adjoin_matrix1, y = x1.to(device), adjoin_matrix1.to(device), y.to(device)
            x2, adjoin_matrix2, index = x2.to(device), adjoin_matrix2.to(device), index.to(device)
            
            t1, t2, t3, t4 = extract_tensors(index, Heavy_dict, Light_dict, Antigen_dict, DAR_dict)
            t4 = t4.view(-1, 1) if t4.dim() == 1 else t4
            seq1 = (x1 == 0).float()
            mask1 = (x1 == 0).unsqueeze(1).unsqueeze(2)
            mask1 = mask1.expand(-1, num_heads, x1.size(1), -1)
            seq2 = (x2 == 0).float()
            mask2 = seq2.unsqueeze(1).unsqueeze(2)
            mask2 = mask2.expand(-1, num_heads, x1.size(1), -1)
            
            optimizer.zero_grad()
            preds = model(x1=x1, mask1=mask1, adjoin_matrix1=adjoin_matrix1, 
                         x2=x2, mask2=mask2, adjoin_matrix2=adjoin_matrix2,
                         t1=t1, t2=t2, t3=t3, t4=t4)
            
            loss = criterion(preds, y.float().squeeze(-1))
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches
        print(f'epoch: {epoch}, loss: {avg_loss:.4f}')

        # Validation
        model.eval()
        y_true = []
        y_preds = []
        
        with torch.no_grad():
            for x1, adjoin_matrix1, y, x2, adjoin_matrix2, index in val_dataset:
                # Move to device
                x1, adjoin_matrix1, y = x1.to(device), adjoin_matrix1.to(device), y.to(device)
                x2, adjoin_matrix2, index = x2.to(device), adjoin_matrix2.to(device), index.to(device)
                
                t1, t2, t3, t4 = extract_tensors(index, Heavy_dict, Light_dict, Antigen_dict, DAR_dict)
                t4 = t4.view(-1, 1) if t4.dim() == 1 else t4
                seq1 = (x1 == 0).float()
                mask1 = (x1 == 0).unsqueeze(1).unsqueeze(2)
                mask1 = mask1.expand(-1, num_heads, x1.size(1), -1)
                seq2 = (x2 == 0).float()
                mask2 = seq2.unsqueeze(1).unsqueeze(2)
                mask2 = mask2.expand(-1, num_heads, x1.size(1), -1)
                
                preds = model(x1=x1, mask1=mask1, adjoin_matrix1=adjoin_matrix1,
                             x2=x2, mask2=mask2, adjoin_matrix2=adjoin_matrix2,
                             t1=t1, t2=t2, t3=t3, t4=t4)
                
                y_true.append(y.cpu())
                y_preds.append(preds.cpu())
        
        y_true = torch.cat(y_true, dim=0).numpy().reshape(-1)
        y_preds = torch.cat(y_preds, dim=0).numpy().reshape(-1)
        y_preds = torch.sigmoid(torch.tensor(y_preds)).numpy()
        auc_new = roc_auc_score(y_true, y_preds)

        print(f'val auc: {auc_new:.4f}')
        
        if auc_new > best_auc:
            best_auc = auc_new
            stopping_monitor = 0
            torch.save(model.state_dict(), f'classification_weights/{task}_{seed}.pth')
            print('save model weights')
        else:
            stopping_monitor += 1
            
        print(f'best val auc: {best_auc:.4f}')
        if stopping_monitor > 0:
            print('stopping_monitor:', stopping_monitor)
        if stopping_monitor > 30:
            break

    # Test evaluation
    model.load_state_dict(torch.load(f'classification_weights/{task}_{seed}.pth', map_location=device))
    model.eval()
    
    y_true = []
    y_preds = []
    
    with torch.no_grad():
        for x1, adjoin_matrix1, y, x2, adjoin_matrix2, index in test_dataset:
            # Move to device
            x1, adjoin_matrix1, y = x1.to(device), adjoin_matrix1.to(device), y.to(device)
            x2, adjoin_matrix2, index = x2.to(device), adjoin_matrix2.to(device), index.to(device)
            
            t1, t2, t3, t4 = extract_tensors(index, Heavy_dict, Light_dict, Antigen_dict, DAR_dict)
            t4 = t4.view(-1, 1) if t4.dim() == 1 else t4
            seq1 = (x1 == 0).float()
            mask1 = (x1 == 0).unsqueeze(1).unsqueeze(2)
            mask1 = mask1.expand(-1, num_heads, x1.size(1), -1)
            seq2 = (x2 == 0).float()
            mask2 = seq2.unsqueeze(1).unsqueeze(2)
            mask2 = mask2.expand(-1, num_heads, x1.size(1), -1)
            
            preds = model(x1=x1, mask1=mask1, adjoin_matrix1=adjoin_matrix1,
                         x2=x2, mask2=mask2, adjoin_matrix2=adjoin_matrix2,
                         t1=t1, t2=t2, t3=t3, t4=t4)
            
            y_true.append(y.cpu())
            y_preds.append(preds.cpu())

    y_true = torch.cat(y_true, dim=0).numpy().reshape(-1)
    y_preds = torch.cat(y_preds, dim=0).numpy().reshape(-1)
    y_preds = torch.sigmoid(torch.tensor(y_preds)).numpy()
    test_auc = roc_auc_score(y_true, y_preds)
    
    tp, tn, fn, fp, se, sp, mcc, acc, auc_roc_score, F1, BA, prauc, PPV, NPV = score(y_true, y_preds)
    print(f'test auc: {test_auc:.4f}')

    return test_auc, tp, tn, fn, fp, se, sp, mcc, acc, auc_roc_score, F1, BA, prauc, PPV, NPV

space = {"dense_dropout": hp.quniform("dense_dropout", 0, 0.5, 0.05), 
        "learning_rate": hp.loguniform("learning_rate", np.log(3e-5), np.log(15e-5)),
        "batch_size": hp.choice("batch_size", [16, 32, 48, 64]),
        "num_heads": hp.choice("num_heads", [4, 8]),
        }

# Hyperparametric search (commented out as in original)
# def hy_main(args):
#     test_auc_list = []
#     x = 0
#     for seed in [2, 8, 9]:
#         print(seed)
#         test_auc,tp, tn, fn, fp, se, sp, mcc, acc, auc_roc_score, F1, BA, prauc, PPV, NPV = main(seed, args)
#         test_auc_list.append(test_auc)
#         x+= test_auc
#     test_auc_list.append(np.mean(test_auc_list))
#     print(test_auc_list)
#     print(args["dense_dropout"])
#     print(args["learning_rate"])
#     print(args["batch_size"])
#     print(args["num_heads"])
#     return -x/3

# best = fmin(hy_main, space, algo = tpe.suggest, max_evals= 30)
# print(best)

best_dict = {}
best_dict["dense_dropout"] = 0.30000000000000004
best_dict["learning_rate"] = 5.5847758199523973e-05
best_dict["batch_size"] = 32
best_dict["num_heads"] = 8
print(best_dict)

if __name__ == '__main__':
    test_auc_list = []
    tp_l, tn_l, fn_l, fp_l, se_l, sp_l, mcc_l, acc_l, auc_roc_score_l, F1_l, BA_l, prauc_l, PPV_l, NPV_l = [],[],[],[],[],[],[],[],[],[],[],[],[],[]
    lists_to_process = [tp_l, tn_l, fn_l, fp_l, se_l, sp_l, mcc_l, acc_l, auc_roc_score_l, F1_l, BA_l, prauc_l, PPV_l, NPV_l]
    
    for seed in [2, 8, 9]:
        print(seed)
        test_auc, tp, tn, fn, fp, se, sp, mcc, acc, auc_roc_score, F1, BA, prauc, PPV, NPV = main(seed, best_dict)
        test_auc_list.append(test_auc)
        tp_l.append(tp)
        tn_l.append(tn)
        fn_l.append(fn)
        fp_l.append(fp)
        se_l.append(se)
        sp_l.append(sp)
        mcc_l.append(mcc)
        acc_l.append(acc)
        auc_roc_score_l.append(auc_roc_score)
        F1_l.append(F1)
        BA_l.append(BA)
        prauc_l.append(prauc)
        PPV_l.append(PPV)
        NPV_l.append(NPV)
    
    test_auc_list.append(np.mean(test_auc_list))
    tp_l.append(np.mean(tp_l))
    tn_l.append(np.mean(tn_l))
    fn_l.append(np.mean(fn_l))
    fp_l.append(np.mean(fp_l))
    se_l.append(np.mean(se_l))
    sp_l.append(np.mean(sp_l))
    mcc_l.append(np.mean(mcc_l))
    acc_l.append(np.mean(acc_l))
    auc_roc_score_l.append(np.mean(auc_roc_score_l))
    F1_l.append(np.mean(F1_l))
    BA_l.append(np.mean(BA_l))
    prauc_l.append(np.mean(prauc_l))
    PPV_l.append(np.mean(PPV_l))
    NPV_l.append(np.mean(NPV_l))
    
    for i in range(len(lists_to_process)):
        lists_to_process[i] = process_list(lists_to_process[i])
    
    filename = 'ADCNet_output.csv'
    column_names = ['tp', 'tn', 'fn', 'fp', 'se', 'sp', 'mcc', 'acc', 'auc', 'F1', 'BA', 'prauc','PPV', 'NPV']
    rows = zip(tp_l, tn_l, fn_l, fp_l, se_l, sp_l, mcc_l, acc_l, auc_roc_score_l, F1_l, BA_l, prauc_l, PPV_l, NPV_l)
    
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(column_names)
        writer.writerows(rows)
    print(f'CSV file {filename} was successfully written')