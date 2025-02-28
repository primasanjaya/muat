import torch
import torch.nn as nn
import torch.optim as optim
import os
import pdb
from muat.util import get_sample_name
import traceback
import pandas as pd
from muat.util import *

class PredictorConfig:
    # optimization parameters
    max_epochs = 1
    batch_size = 1
    result_dir = None
    target_handler = None
    get_features = False

    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)

class Predictor:

    def __init__(self, model, test_dataset, config):
        self.model = model
        self.test_dataset = test_dataset
        self.config = config

        # take over whatever gpus are on the system
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()

        self.result_dir = ensure_dirpath(self.config.result_dir)

    def batch_predict(self):
        model = self.model
        model = model.to(self.device)

        model = torch.nn.DataParallel(model).to(self.device)
        model.train(False)
        #pdb.set_trace()

        ensure_dir_exists(self.result_dir)

        for i in range(len(self.test_dataset)):
            data, target, sample_path = self.test_dataset.__getitem__(i)
            numeric_data = data.unsqueeze(0)
            numeric_data = numeric_data.to(self.device)
            # forward the model 
            with torch.set_grad_enabled(False):
                logits, _ = model(numeric_data)
                #pdb.set_trace()
                if isinstance(logits, dict):
                    #write all logits
                    #pdb.set_trace()
                    logit_keys = [x for x in logits.keys() if 'logits' in x]
                    for nk, lk in enumerate(logit_keys):
                        logit = logits[lk]
                        _, predicted = torch.max(logit.data, 1)
                        predicted_cpu = predicted.detach().cpu().numpy().flatten()

                        target_handler = self.config.target_handler[nk]
                        target_name = target_handler.inverse_transform(predicted_cpu)[0]

                        logits_cpu =logit.detach().cpu().numpy()
                        logit_filename = 'prediction_{}.tsv'.format(lk)
                        
                        if i==0:
                            f = open(self.result_dir + logit_filename, 'w+') 
                            header_class = target_handler.classes_
                            header_class.append('prediction')
                            header_class.append('sample')
                            write_header = "\t".join(header_class)
                            f.write(write_header)
                            f.close()
                        else:
                            #write logits
                            f = open(self.result_dir + logit_filename, 'a+')
                            logits_cpu =logit.detach().cpu().numpy()
                            f.write('\n')
                            logits_cpu_flat = logits_cpu.flatten()
                            logits_cpu_list = logits_cpu_flat.tolist()
                            write_logits = [f'{i:.8f}' for i in logits_cpu_list]
                            write_logits.append(str(target_name))
                            write_logits.append(get_sample_name(sample_path))
                            write_header = "\t".join(write_logits)
                
                            f.write(write_header)
                            f.close()
                        if nk == 0:
                            print(get_sample_name(sample_path) + ' is predicted to be ' + str(target_name))

                    #write all features
                    feature_keys = [x for x in logits.keys() if 'features' in x]
                    for nk, lk in enumerate(feature_keys):
                        feat = logits[lk]
                        feat_cpu = feat.detach().cpu().numpy()
                        feat_filename = 'features_{}.tsv'.format(lk)
                        
                        if i==0:
                            f = open(self.result_dir + feat_filename, 'w+') 
                            feat_cpu_flat = feat_cpu.flatten()
                            feat_cpu_list = feat_cpu_flat.tolist()
                            write_header = [f'M{i+1}' for i in range(len(feat_cpu_list))]
                            write_header.append('sample')
                            write_header = "\t".join(write_header)
                            f.write(write_header)
                            f.close()
                        else:
                            #write features
                            f = open(self.result_dir + feat_filename, 'a+')
                            f.write('\n')
                            feat_cpu_flat = feat_cpu.flatten()
                            feat_cpu_list = feat_cpu_flat.tolist()
                            write_feat = [f'{i:.8f}' for i in feat_cpu_list]
                            write_feat.append(get_sample_name(sample_path))
                            write_feat = "\t".join(write_feat)
                            f.write(write_feat)
                            f.close()
        print('Results have been saved in ' + self.result_dir)