import os
import sys
import pickle
import lmdb
import pandas as pd
import numpy as np
from rdkit import Chem
from tqdm import tqdm
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import RDLogger
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
RDLogger.DisableLog('rdApp.*')  
import warnings
warnings.filterwarnings(action='ignore')
from multiprocessing import Pool
from sklearn.model_selection import train_test_split
import os
import sys
from functools import partial
# sys.path.append("/usr/local/lib/python3.10/dist-packages/lmdb-1.4.1-py3.10-linux-x86_64.egg")
# os.chdir("/content")
def smi2scaffold(smi):
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(
            smiles=smi, includeChirality=True)
    except:
        print("failed to generate scaffold with smiles: {}".format(smi))
        return smi

def smi2_2Dcoords(smi:str):
    mol = Chem.MolFromSmiles(smi)
    mol = AllChem.AddHs(mol)
    AllChem.Compute2DCoords(mol)
    coordinates = mol.GetConformer().GetPositions().astype(np.float32)
    len(mol.GetAtoms()) == len(coordinates), "2D coordinates shape is not align with {}".format(smi)
    return coordinates

def smi2_3Dcoords(smi:str,cnt:int=10,seed:int=42):
    mol = Chem.MolFromSmiles(smi)
    mol = AllChem.AddHs(mol)
    coordinate_list=[]
    for seed in range(cnt):
        try:
            res = AllChem.EmbedMolecule(mol, randomSeed=seed)  # will random generate conformer with seed equal to -1. else fixed random seed.
            if res == 0:
                try:
                    AllChem.MMFFOptimizeMolecule(mol)       # some conformer can not use MMFF optimize
                    coordinates = mol.GetConformer().GetPositions()
                except:
                    print("Failed to generate 3D, replace with 2D")
                    coordinates = smi2_2Dcoords(smi)            
                    
            elif res == -1:
                mol_tmp = Chem.MolFromSmiles(smi)
                AllChem.EmbedMolecule(mol_tmp, maxAttempts=5000, randomSeed=seed)
                mol_tmp = AllChem.AddHs(mol_tmp, addCoords=True)
                try:
                    AllChem.MMFFOptimizeMolecule(mol_tmp)       # some conformer can not use MMFF optimize
                    coordinates = mol_tmp.GetConformer().GetPositions()
                except:
                    print("Failed to generate 3D, replace with 2D")
                    coordinates = smi2_2Dcoords(smi) 
        except:
            print("Failed to generate 3D, replace with 2D")
            coordinates = smi2_2Dcoords(smi) 

        assert len(mol.GetAtoms()) == len(coordinates), "3D coordinates shape is not align with {}".format(smi)
        coordinate_list.append(coordinates.astype(np.float32))
    return coordinate_list

def inner_smi2coords(content:tuple,seed:int=42):
    smi = content[0]
    target = content[1:]
    cnt = 10 # conformer num,all==11, 10 3d + 1 2d
    scaffold = smi2scaffold(smi)

    mol = Chem.MolFromSmiles(smi)
    if len(mol.GetAtoms()) > 400:
        coordinate_list =  [smi2_2Dcoords(smi)] * (cnt+1)
        print("atom num >400,use 2D coords",smi)
    else:
        coordinate_list = smi2_3Dcoords(smi,cnt,seed)
        coordinate_list.append(smi2_2Dcoords(smi).astype(np.float32))
    mol = AllChem.AddHs(mol)
    atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]  # after add H 
    return pickle.dumps({'atoms': atoms, 
    'coordinates': coordinate_list, 
    'mol':mol,'smi': smi, 'scaffold': scaffold, 'target': target}, protocol=-1)

def smi2coords(content:tuple,seed:int=42):
    try:
        return inner_smi2coords(content,seed)
    except:
        print("failed smiles: {}".format(content[0]))
        return None

def write_lmdb(inpath: str='./', outpath: str='./', nthreads:int=16, seed:int=42):
    print("Generate lmdb data for Uni-Mol model")

    train = pd.read_csv(os.path.join(inpath,'train.csv'))
    valid = pd.read_csv(os.path.join(inpath,'valid.csv'))
    test = pd.read_csv(os.path.join(inpath,'test.csv'))
   
    for name, content_list in [('train.lmdb', zip(*[train[c].values.tolist() for c in train])),
                                ('valid.lmdb', zip(*[valid[c].values.tolist() for c in valid])),
                                ('test.lmdb', zip(*[test[c].values.tolist() for c in test]))]:
    # for name, content_list in [('test.lmdb', zip(*[test[c].values.tolist() for c in test]))]:
        if not os.path.exists(outpath):
          os.mkdir(outpath)
        output_name = os.path.join(outpath, name)
        try:
            os.remove(output_name)
        except:
            pass
        env_new = lmdb.open(
            output_name,
            subdir=False,
            readonly=False,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=1,
            map_size=int(100e9),
        )
        txn_write = env_new.begin(write=True)
        with Pool(nthreads) as pool:
            i = 0
            for inner_output in tqdm(pool.imap(smi2coords, content_list)):
                if inner_output is not None:
                    txn_write.put(f'{i}'.encode("ascii"), inner_output)
                    i += 1
            print('{} process {} lines'.format(name, i))
            txn_write.commit()
            env_new.close()
            
def one_csv_to_lmdb(data_path:str, seed:int=42):
    '''
    csv colname is highly tolerant:
    first col is the smiles and 
    other cols will be the property to be predict. 
    '''
    assert data_path.endswith('.csv'),'only accept .csv file!'
    datadir=data_path.replace('.csv','')
    os.makedirs(datadir,exist_ok=True)
    # assert not os.path.exists(datadir),(f'{datadir} already exists!\n'
    #         ' remove the directory or skip data-preprocessing.')
    data = pd.read_csv(data_path)
    assert ['']
    op = lambda x:os.path.join(datadir,x)
    train, val_test = train_test_split(data, test_size=0.2, random_state = seed)
    valid, test = train_test_split(val_test, test_size=0.5, random_state = seed)
    train.to_csv(op('train.csv'), index=False)
    valid.to_csv(op('valid.csv'), index=False)
    test.to_csv(op('test.csv'), index=False)
    write_lmdb(datadir,datadir,seed=seed)
    
    
def three_csv_to_lmdb(data_path:str, seed:int=42):
    '''
    data_path: dir contains {train,valid,test}.csv
    '''
    op = lambda x:os.path.join(data_path,x)
    for i in ['train','valid','test']:
        f=op(f'{i}.csv')
        assert os.path.isfile(f),(f'missing required file:\n{f}')
    write_lmdb(data_path,data_path,seed=seed)

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--data',required=True,help=('data to process.\n'
        'it can be:\n 1)a .csv file, \n'
        'whose first col is smiles and other cols are target to learn.\n'
        '2) a directory contains {train,valid,test}.csv,\n'
        ' with same rule described above'))
    parser.add_argument('--seed',required=False,default=42,
        type=int,help=('(optional) seed for split file and generate 3d conformation.'
                       'default=42'))
    args=parser.parse_args()
    data_path:str=args.data
    seed:int=args.seed
    if data_path.endswith('.csv'):
        one_csv_to_lmdb(data_path,seed)
    else:
        three_csv_to_lmdb(data_path,seed)
        
#   write_lmdb(inpath='/content/data', outpath='/content/data', nthreads=16)