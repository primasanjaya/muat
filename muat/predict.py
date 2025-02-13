import torch
import torch.nn as nn
import torch.optim as optim
import os
import pdb
from muat.util import get_sample_name
import traceback
import pandas as pd

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
        self.global_acc = 0
        self.pd_logits = []

        # take over whatever gpus are on the system
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()

        self.result_dir = self.config.result_dir

    def batch_predict(self):
        model, config = self.model, self.config
        model = model.to(self.device)

        model = torch.nn.DataParallel(model).to(self.device)
        batch_size = self.config.batch_size

        valloader = torch.utils.data.DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False)

        #val
        test_loss = 0

        self.logit_filename = 'prediction_logits.tsv'
        #pdb.set_trace()
        f = open(self.result_dir + '/' + self.logit_filename, 'w+')  # open file in write mode

        header_class = config.target_handler.classes_.tolist()
        header_class.append('prediction')
        header_class.append('sample')
        write_header = "\t".join(header_class)
        f.write(write_header)
        f.close()

        if config.get_features:
            self.feature_filename = 'prediction_features.tsv'
            f = open(self.result_dir + '/' + self.feature_filename, 'w+')  # open file in write mode
            f.close()
            write_header_feature = False
        
        model.train(False)

        for i in range(len(self.test_dataset)):
            data, target, sample_path = self.test_dataset.__getitem__(i)
            numeric_data = data.unsqueeze(0)
            numeric_data = numeric_data.to(self.device)
            # forward the model 
            with torch.set_grad_enabled(False):
                logits, loss = model(numeric_data)
                _, predicted = torch.max(logits.data, 1)

                predicted = logits.argmax(dim=1, keepdim=True)  # get the index of the max log-probability

                #write logits
                 #pdb.set_trace()
                f = open(self.result_dir + '/' + self.logit_filename, 'a+')
                logits_cpu =logits.detach().cpu().numpy()
                f.write('\n')
                #pdb.set_trace()
                logits_cpu_flat = logits_cpu.flatten()
                logits_cpu_list = logits_cpu_flat.tolist()
                write_logits = ["%.8f" % i for i in logits_cpu_list]
                predicted_cpu = predicted.detach().cpu().numpy().flatten()
                target_name = self.config.target_handler.inverse_transform(predicted_cpu)[0]
                write_logits.append(str(target_name))
                write_logits.append(get_sample_name(sample_path))
                write_header = "\t".join(write_logits)
     
                f.write(write_header)
                f.close()

                print(get_sample_name(sample_path) + ' is predicted as ' + str(target_name))

                if config.get_features:
                    try:
                        features = model(numeric_data,get_features=True)
                        features = features.detach().cpu().numpy()[0]

                        pd_cpu_features = pd.DataFrame(features).T

                        featurecolumn = []
                        count = 0

                        for i in range(len(pd_cpu_features.columns)):
                            count = count + 1
                            featurecolumn.append('M' + str(count))

                        pd_cpu_features.columns = featurecolumn

                        f = open(self.result_dir + '/' + self.feature_filename, 'a+')

                        if write_header_feature == False:
                            header_class = featurecolumn
                            write_header = "\t".join(header_class)
                            write_header=write_header+"\tsamples"
                            f.write(write_header)
                            write_header_feature = True

                        f.write('\n')
                        logits_cpu_list = pd_cpu_features.iloc[0].values
                        #print(logits_cpu_list) 
                        write_logits = ["%.8f" % i for i in logits_cpu_list]
                        write_logits = "\t".join(write_logits)
                        write_logits=write_logits+"\t"+get_sample_name(sample_path)
                        f.write(write_logits)
                        f.close()

                    except Exception as e:
                        # Print the exception message
                        print(f"An error occurred: {e}")
                        # Get the complete traceback information
                        traceback_info = traceback.format_exc()
                        print("Traceback details:")
                        print(traceback_info)