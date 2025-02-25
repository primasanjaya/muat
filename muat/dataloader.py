import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from torch.utils.data import Dataset
import os
import pandas as pd
import pdb
import numpy as np
import math
import pickle
import random
from sklearn.utils import shuffle

class DataloaderConfig:

    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)

class MuAtDataloader(Dataset):
    def __init__(self, data_split_tsv,config,same_sampling=False):
        self.data_split_tsv = data_split_tsv
        self.model_input = config.model_input
        self.mutation_type = config.mutation_type
        self.mutation_sampling_size = config.mutation_sampling_size
        self.same_sampling = same_sampling

    def __len__(self):
        return len(self.data_split_tsv)
    
    def __getitem__(self, idx):
        return self.get_data(idx)

    def count_ratio(self,pd_row):
        row_count_init = {'SNV':0,'MNV':0,'indel':0,'SV/MEI':0,'Neg':0}
        count = pd_row.groupby('mut_type').size().to_dict()
        for key,value in count.items():
            row_count_init[key] = value
            if key == 'SV':
                row_count_init['SV/MEI'] += value
            elif key == 'MEI':
                row_count_init['SV/MEI'] += value

        mut_ratio = np.array(list(self.mutation_type.values()))
        avail_count = mut_ratio * self.mutation_sampling_size   
        row_count = np.array(list(row_count_init.values()))
            
        diff = avail_count - row_count
        pos = diff>0
        avail_count1 = row_count * pos
        diff = row_count > avail_count

        avail_count2 = avail_count * diff
        avail_count3 = avail_count1 + avail_count2
        shadowavail_count3 = avail_count3
        shadowavail_count3[0] = row_count[0]

        if sum(shadowavail_count3) > self.mutation_sampling_size:
            diff = self.mutation_sampling_size - sum(avail_count3) 
            shadowavail_count3[0] = diff + avail_count3[0]
            
        avail_count2 = shadowavail_count3.astype(int)

        if avail_count2[0]<0:

            secondmax = avail_count2[np.argmax(avail_count2)]
            avail_count2 = avail_count2 * 0.7

            avail_count = avail_count2

            diff = avail_count - row_count
            pos = diff>0
            avail_count1 = row_count * pos
            diff = row_count > avail_count

            avail_count2 = avail_count * diff
            avail_count3 = avail_count1 + avail_count2
            shadowavail_count3 = avail_count3
            shadowavail_count3[0] = row_count[0]

            if sum(shadowavail_count3) > self.mutation_sampling_size:
                diff = self.mutation_sampling_size - sum(avail_count3) 
                shadowavail_count3[0] = diff + avail_count3[0]
                
            avail_count2 = shadowavail_count3.astype(int)

        avail_count = avail_count2

        avail_count_dict = {}

        for i,key in enumerate(row_count_init.keys()):
            avail_count_dict[key] = avail_count[i]

        return avail_count_dict

    def get_data(self, idx):
        instances = self.data_split_tsv.iloc[idx]
        pd_row = pd.read_csv(instances['prep_path'], sep='\t', compression='gzip', low_memory=False)
        sample_path = instances['prep_path']
        
        # Get idx_class and idx_subclass if they exist
        idx_class = None
        if 'idx_class' in instances.index.to_list():
            idx_class = torch.tensor(np.array(instances['idx_class']), dtype=torch.long)
        
        idx_subclass = None
        if 'idx_subclass' in instances.index.to_list():
            idx_subclass = torch.tensor(np.array(instances['idx_subclass']), dtype=torch.long)
        
        # Sampling logic
        avail_count = self.count_ratio(pd_row)
        pd_sampling = pd.DataFrame()
        grab_col = []

        if self.model_input['motif']:
            grab_col.append('triplettoken')
        if self.model_input['pos']:
            grab_col.append('postoken')
        if self.model_input['ges']:
            grab_col.append('gestoken')

        for key, value in avail_count.items():
            if value > 0:
                pd_samp = pd_row[pd_row['mut_type'] == key][grab_col].sample(n=value, replace=False)
                pd_sampling = pd.concat([pd_sampling, pd_samp], ignore_index=True)
        
        # Handle padding
        np_triplettoken = pd_sampling.to_numpy()
        is_padding = len(pd_sampling) < self.mutation_sampling_size
        mins = self.mutation_sampling_size - len(np_triplettoken) if is_padding else 0

        datanumeric = []
        for col in pd_sampling.columns:
            np_data = pd_sampling[col].to_numpy()
            if is_padding:
                np_data = np.pad(np_data, (0, mins), mode='constant', constant_values=0)
            np_data = np.asarray(np_data[:self.mutation_sampling_size], dtype=int)
            datanumeric.append(torch.tensor(np_data, dtype=torch.long))

        # Ensure datanumeric is valid
        datanumeric = torch.stack(datanumeric)

        # Ensure no None values in data_targets
        data_targets = {
            "idx_class": idx_class if idx_class is not None else [],
            "idx_subclass": idx_subclass if idx_subclass is not None else []
        }

        return datanumeric, data_targets, sample_path




if __name__ == '__main__':

    #dataloader = PCAWG(dataset_name = 'PCAWG', data_dir='/csc/epitkane/projects/PCAWG/shuffled_samples/', mode='training',portion = [8,1,1], folds=10, curr_fold=1,load=True,load_token=True)

    #dataloader = PCAWG(dataset_name = 'pcawg_mut3_comb0', data_dir='/csc/epitkane/projects/PCAWG20191001/data/modified_data/train/all24classes/', mode='training',portion = [8,1,1], folds=10, curr_fold=1,load=True,load_token=True,ncontext=3,addposition=False,filter=False,topk=5000)
    #dataloaderVal = PCAWG(dataset_name = 'pcawg_mut3_comb0', data_dir='/csc/epitkane/projects/PCAWG20191001/data/modified_data/train/all24classes/', mode='validation',portion = [8,1,1], folds=10, curr_fold=1,load=True,load_token=True,ncontext=3,addposition=False,filter=False,topk=5000)
    #/csc/epitkane/projects/tcga/new23classes/
    #/csc/epitkane/projects/PCAWG20191001/data/modified_data/train/new24classes/

    #G:/experiment/data/new24classes/
    '''
    dataloaderVal = FinalTCGAPCAWG(dataset_name = 'finalpcawg', 
                                data_dir='G:/experiment/data/new24classes/', 
                                mode='validation', 
                                curr_fold=1, 
                                block_size=5000, 
                                load=False,
                                mutratio = '0.3-0.3-0.3-0-0',
                                addtriplettoken=False,
                                addpostoken=False,
                                addgestoken=True,
                                addrt=False,
                                nummut = 0,
                                frac = 0,
                                adddatadir='G:/experiment/data/icgc/')

    #pdb.set_trace()
    data,target = dataloaderVal.__getitem__(0)
    pdb.set_trace()

    for k in range(0,len(dataloaderVal)):
        print(k)
        data,target = dataloaderVal.__getitem__(k)
    '''



    '''
    WGS GX
    '''

    #/scratch/project_2001668/data/pcawg

    dataloaderVal = TCGAPCAWG_Dataloader(dataset_name = 'wgsgx', 
                                        data_dir='/scratch/project_2001668/data/pcawg/allclasses/newformat/', 
                                        mode='training', 
                                        curr_fold=1, 
                                        block_size=5000, 
                                        load=False,
                                        addtriplettoken=True,
                                        addpostoken=False,
                                        addgestoken=False,
                                        addrt=False,
                                        nummut = 0,
                                        frac = 0,
                                        mutratio = '1-0-0-0-0',
                                        adddatadir = None,
                                        input_filename=None,
                                        args = None,
                                        gx_dir = '/scratch/project_2001668/data/pcawg/PCAWG_geneexp/')
    
    data,target = dataloaderVal.__getitem__(0)
    pdb.set_trace()

    '''
    fold = [1,2,3,4,5,6,7,8,9,10]
    mutratios = ['1-0-0-0-0','0.5-0.5-0-0-0','0.4-0.3-0.3-0-0','0.3-0.3-0.20-0.20-0','0.25-0.25-0.25-0.15-0.1']

    retrieve = ['addtriplettoken','addpostoken','addgestoken','addrt']

    for fo in fold:
        for i in retrieve:
            if i == 'addtriplettoken':
                addtriplettoken = True
            else:
                addtriplettoken = False
            
            if i == 'addpostoken':
                addpostoken = True
            else:
                addpostoken = False

            if i == 'addgestoken':
                addgestoken = True
            else:
                addgestoken = False

            if i == 'addrt':
                addrt = True
            else:
                addrt = False

            for j in mutratios:
                dataloaderVal = FinalTCGAPCAWG(dataset_name = 'finalpcawg', 
                                    data_dir='G:/experiment/data/new24classes/', 
                                    mode='validation', 
                                    curr_fold=1, 
                                    block_size=5000, 
                                    load=False,
                                    mutratio = j,
                                    addtriplettoken=addtriplettoken,
                                    addpostoken=addpostoken,
                                    addgestoken=addgestoken,
                                    addrt=addrt,
                                    nummut = 0,
                                    frac = 0)
                for k in range(0,len(dataloaderVal)):
                    print(str(fo) + ' ' + str(k) + ' ' + i + ' ' + j + ' ' + str(addtriplettoken) + str(addpostoken) + str(addgestoken) + str(addrt))
                    data,target = dataloaderVal.__getitem__(k)
    pdb.set_trace()

    dataloaderVal = TCGA(dataset_name = 'tcga_emb', data_dir='/csc/epitkane/projects/tcga/all23classes/', mode='validation',portion = [8,1,1], folds=10, curr_fold=1,load=True,load_token=True,ncontext=64,addposition=True,filter=True,block_size=300,withclass=True,twostream=False)

    for i in range(len(dataloaderVal)):
        data,target = dataloaderVal.__getitem__(i)

    dataloaderVal = TCGA(dataset_name = 'tcga_emb', data_dir='/csc/epitkane/projects/tcga/all23classes/', mode='testing',portion = [8,1,1], folds=10, curr_fold=1,load=True,load_token=True,ncontext=64,addposition=True,filter=True,block_size=300,loaddist=False,withclass=True,twostream=False)

    for i in range(len(dataloaderVal)):
        data,target = dataloaderVal.__getitem__(i)
    
    pdb.set_trace()
    '''