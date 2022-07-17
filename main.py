from tdc.multi_pred import DTI
from DeepPurpose import utils, model_helper, dataset
import DeepPurpose.DTI as DT
import pandas as pd
import numpy as np
import wget
from zipfile import ZipFile
from DeepPurpose.utils import *
import json
import os
import dgl

def read_file(path = './data', binary = False, threshold = 9):
    affinity = pd.read_csv(path + '/KIBA/affinity.txt', header=None, sep='\t')
    affinity = affinity.fillna(-1)

    with open(path + '/KIBA/target_seq.txt') as f:
        target = json.load(f)

    with open(path + '/KIBA/SMILES.txt') as f:
        drug = json.load(f)

    target = list(target.values())
    drug = list(drug.values())

    SMILES = []
    Target_seq = []
    y = []

    for i in range(len(drug)):
        for j in range(len(target)):
            if affinity.values[i, j] != -1:
                SMILES.append(drug[i])
                Target_seq.append(target[j])
                y.append(affinity.values[i, j])

    if binary:
        print('Note that KIBA is not suitable for binary classification as it is a modified score. \
    		   Default binary threshold for the binding affinity scores are 9, \
    		   you should adjust it by using the "threshold" parameter')
        y = [1 if i else 0 for i in np.array(y) < threshold]
    else:
        y = y

    print('Done!')
    return np.array(SMILES), np.array(Target_seq), np.array(y)

data = DTI(name = 'KIBA')
split = data.get_split()

X_drugs, X_targets, y = read_file(path='./data', binary = False, threshold = 30)

print('Drug 1: ' + X_drugs[0])
print('Target 1: ' + X_targets[0])
print('Score 1: ' + str(y[0]))

drug_encoding, target_encoding = 'Morgan', 'CNN'

train, val, test = utils.data_process(X_drugs, X_targets, y,
                                drug_encoding, target_encoding,
                                split_method='random',frac=[0.8,0.1,0.1],
                                random_seed = 1)

config = utils.generate_config(drug_encoding = drug_encoding,
                         target_encoding = target_encoding,
                         cls_hidden_dims = [1024,1024,512],
                         train_epoch = 10,
                         LR = 0.01,
                         batch_size = 128,
                         hidden_dim_drug = 128,
                         mpnn_hidden_size = 128,
                         mpnn_depth = 3,
                         cnn_target_filters = [32,64,96],
                         cnn_target_kernels = [4,8,12]
                        )

model = DT.model_initialize(**config)
model.train(train, val, test)
model.save_model('./tutorial_model2')


