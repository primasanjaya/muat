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
import zlib
from sklearn.utils import shuffle


def _sample_seed(*parts):
    """A random_state for pandas .sample() derived only from `parts` (typically the
    run's seed, the sample's own path, and a mutation-type key), so a given
    seed+sample+key always draws the identical subsample regardless of when/how many
    other draws happened first in this process. Plain ambient-RNG .sample() calls are
    call-order dependent -- that's why training's internal validation (deep into an
    epoch loop) and a fresh `predict` call on the same checkpoint used to draw
    different 5000-mutation subsamples for the same sample and disagree, even though
    both are individually reproducible run-to-run. Folding the seed itself in (rather
    than ignoring it) means different seeds genuinely draw different subsamples, so
    an unseeded run and a seeded one -- or two different seeds -- are not silently
    forced onto the same sampling."""
    key = '|'.join(str(p) for p in parts)
    return zlib.crc32(key.encode()) & 0xffffffff

class DataloaderConfig:
    def __init__(
        self,
        model_input=None,
        mutation_type_ratio=None,
        mutation_sampling_size=None,
        sampling_replacement=False,
        **kwargs
    ):
        self.model_input = model_input
        self.mutation_type_ratio = mutation_type_ratio
        self.mutation_sampling_size = mutation_sampling_size
        self.sampling_replacement = sampling_replacement

        for k, v in kwargs.items():
            setattr(self, k, v)

        if self.model_input is None:
            raise ValueError("DataloaderConfig requires model_input.")
        if self.mutation_type_ratio is None:
            raise ValueError("DataloaderConfig requires mutation_type_ratio.")
        if self.mutation_sampling_size is None:
            raise ValueError("DataloaderConfig requires mutation_sampling_size.")

class MuAtDataloader(Dataset):
    def __init__(self, data_split_tsv,config,same_sampling=False):
        self.config = config
        self.data_split_tsv = data_split_tsv
        self.model_input = config.model_input
        self.mutation_type_ratio = config.mutation_type_ratio
        self.mutation_sampling_size = config.mutation_sampling_size
        self.same_sampling = same_sampling
        self.sampling_replacement = getattr(config, 'sampling_replacement', False)

        if 'prep_path' not in data_split_tsv.columns:
            raise ValueError("MuAtDataloader: input dataframe is missing required column 'prep_path'.")
        missing = [p for p in data_split_tsv['prep_path'].tolist() if not os.path.exists(p)]
        if missing:
            preview = '\n  '.join(missing[:5])
            raise FileNotFoundError(
                "MuAtDataloader: {} input file(s) do not exist. First missing:\n  {}".format(
                    len(missing), preview))

    def __len__(self):
        return len(self.data_split_tsv)
    
    def __getitem__(self, idx):
        return self.get_data(idx)

    def count_ratio(self,pd_row):
        row_count_init = {'SNV':0,'MNV':0,'indel':0,'SV/MEI':0,'Neg':0}
        count = pd_row.groupby('mut_type').size().to_dict()
        # mut_type values are the literal strings 'SV'/'MEI' (see reader.py's
        # Variant.SV_TYPES/MEI_TYPES), not 'SV/MEI' -- assigning row_count_init[key]
        # unconditionally for those two would silently ADD stray extra dict keys
        # ('SV', 'MEI') alongside the intended 'SV/MEI' bucket, growing this dict from
        # 5 to 6 entries and breaking the array arithmetic below. Never triggered before
        # because no preprocessed data had real SV/MEI mut_type values until the
        # d1_snvmnvindelsv tag.
        for key,value in count.items():
            if key in ('SV', 'MEI'):
                row_count_init['SV/MEI'] += value
            else:
                row_count_init[key] = value

        mut_ratio = np.array(list(self.mutation_type_ratio.values()))
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
        if instances['prep_path'].endswith('.gz'):
            pd_row = pd.read_csv(instances['prep_path'], sep='\t', compression='gzip', low_memory=False)
        else:
            pd_row = pd.read_csv(instances['prep_path'], sep='\t', low_memory=False)

        sample_path = instances['prep_path']
        
        # Get idx_class and idx_subclass if they exist
        idx_class = None
        if 'class_index' in instances.index.to_list():
            idx_class = torch.tensor(np.array(instances['class_index']), dtype=torch.long)
        
        idx_subclass = None
        if 'subclass_index' in instances.index.to_list():
            idx_subclass = torch.tensor(np.array(instances['subclass_index']), dtype=torch.long)
        
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

        # A seed on the config makes sampling deterministic per (seed, sample, key) --
        # reproducible across processes (e.g. predict matching training's own
        # validation draw for the same checkpoint). No seed (unseeded runs) means
        # genuinely random: draws from pandas' ambient RNG state as before, varying
        # call to call, exactly like an unseeded run should.
        run_seed = getattr(self.config, 'seed', None)

        for key, value in avail_count.items():
            if value > 0:
                # 'SV/MEI' is a combined bucket (see count_ratio()) -- the mut_type
                # column itself never contains that literal string, only 'SV'/'MEI'
                # separately (reader.py's Variant.SV_TYPES/MEI_TYPES), so an exact-match
                # filter here would silently select zero rows and crash .sample() below
                # with "a must be greater than 0". Never triggered before because no
                # preprocessed data had real SV/MEI mut_type values until this tag.
                if key == 'SV/MEI':
                    row_mask = pd_row['mut_type'].isin(['SV', 'MEI'])
                else:
                    row_mask = pd_row['mut_type'] == key
                subset = pd_row[row_mask][grab_col]
                if run_seed is not None:
                    pd_samp = subset.sample(n=value, replace=False,
                                             random_state=_sample_seed(run_seed, sample_path, key))
                else:
                    pd_samp = subset.sample(n=value, replace=False)
                pd_sampling = pd.concat([pd_sampling, pd_samp], ignore_index=True)

        # Handle padding
        if self.sampling_replacement:
            np_triplettoken = pd_sampling.to_numpy()
            mins = self.mutation_sampling_size - len(np_triplettoken)
            if run_seed is not None:
                pd_rest_sampling = pd_sampling.sample(
                    n=mins, replace=True, random_state=_sample_seed(run_seed, sample_path, 'pad'))
            else:
                pd_rest_sampling = pd_sampling.sample(n=mins, replace=True)
            pd_sampling = pd.concat([pd_sampling, pd_rest_sampling], ignore_index=True)
            datanumeric = torch.tensor(pd_sampling.to_numpy().T, dtype=torch.long)
        else:
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
            "class_index": idx_class if idx_class is not None else [],
            "subclass_index": idx_subclass if idx_subclass is not None else []
        }

        return datanumeric, data_targets, sample_path