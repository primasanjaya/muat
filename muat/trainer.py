import torch
import torch.optim as optim
import torch.utils.data
import os
import shutil
import pdb
import logging
import numpy as np
from muat.util import *
import json
import zipfile
import glob
import pandas as pd

logger = logging.getLogger(__name__)

class TrainerConfig:
    # optimization parameters
    max_epochs = 10
    batch_size = 4
    learning_rate = 3e-4
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.001 # only applied on matmul weights
    # learning rate decay params: linear warmup followed by cosine decay to 10% of original
    lr_decay = False

    show_loss_interval = 10

    # checkpoint settings
    save_ckpt_path = None
    string_logs = None
    num_workers = 0 # for DataLoader
    ckpt_name = 'model'
    args = None

    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)

class Trainer:

    def __init__(self, model, train_dataset, test_dataset, config):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.config = config
        self.global_acc = 0
        self.pd_logits = []

        # take over whatever gpus are on the system
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
        
        self.complete_save_dir = self.config.save_ckpt_dir

    def batch_train(self):
        model = self.model
        model = model.to(self.device)

        if self.config.save_ckpt_dir is not None:
            os.makedirs(self.config.save_ckpt_dir, exist_ok=True) 

        model = torch.nn.DataParallel(model).to(self.device)
        optimizer = optim.SGD(model.parameters(), lr=self.config.learning_rate, momentum=0.9,weight_decay=self.config.weight_decay)

        trainloader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.config.batch_size, shuffle=True)
        valloader = torch.utils.data.DataLoader(self.test_dataset, batch_size=self.config.batch_size, shuffle=False)

        self.global_acc = 0
        self.save_checkpoint_v3(self.config.save_ckpt_dir)

        for e in range(self.config.max_epochs):
            running_loss = 0
            model.train(True)

            train_corr = []

            for batch_idx, (data, target, sample_path) in enumerate(trainloader):

                string_data = target
                numeric_data = data
                numeric_data = numeric_data.to(self.device)
                class_keys = [x for x in string_data.values() if not isinstance(x, list)]
                class_values = []
                for x in string_data.keys():
                    values = string_data[x]
                    if not isinstance(values, list):
                        class_values.append(values)
                #pdb.set_trace()
                if len(class_values)>1:
                    target = torch.stack(class_values, dim=0)
                elif len(class_values)==1:
                    target = class_values[0].unsqueeze(dim=0)
                target.to(self.device)

                # forward the model
                with torch.set_grad_enabled(True):

                    optimizer.zero_grad()
                    #pdb.set_trace()
                    logits, loss = model(numeric_data, target)

                    if isinstance(logits, dict):
                        logit_keys = [x for x in logits.keys() if 'logits' in x]

                        train_corr_inside = []

                        for nk, lk in enumerate(logit_keys):
                            logit = logits[lk]
                            _, pred = torch.max(logit.data, 1)                            
                            logits_cpu =logit.detach().cpu().numpy()
                            pred = logit.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                            # Ensure target is on the same device as pred
                            target_on_device = target[nk].to(pred.device)  # Move target to the same device as pred
                            train_corr_inside.append(pred.eq(target_on_device.view_as(pred)).sum().item())

                        if len(train_corr) == 0:
                            train_corr = np.zeros(len(logit_keys))
                        train_corr += np.asarray(train_corr_inside)
                    else:
                        pass                   
                    
                    loss.backward()
                    #And optimizes its weights here
                    optimizer.step()
                    running_loss += loss.item()
                    train_acc = train_corr / len(self.train_dataset)

                    if batch_idx % self.config.show_loss_interval == 0:
                        show_text = "Epoch {} - Batch ({}/{}) - Mini-batch Training loss: {:.4f}".format(e+1,batch_idx , len(trainloader) , running_loss/(batch_idx+1))
                        for x in range(len(logit_keys)):
                            show_text = show_text + ' - Training Acc {}: {:.2f}'.format(x+1,train_acc[x])
                        print(show_text)
            show_text = "Epoch {} - Full-batch Training loss: {:.4f}".format(e+1, running_loss/(batch_idx+1))
            for x in range(len(logit_keys)):
                show_text = show_text + ' - Training Acc {}: {:.2f}'.format(x+1,train_acc[x])
            print(show_text)

            #validation
            test_loss = 0
            test_correct = []
            
            model.train(False)
            for batch_idx_val, (data, target, sample_path) in enumerate(valloader):

                string_data = target
                numeric_data = data
                numeric_data = numeric_data.to(self.device)
                class_keys = [x for x in string_data.values() if not isinstance(x, list)]
                class_values = []
                for x in string_data.keys():
                    values = string_data[x]
                    if not isinstance(values, list):
                        class_values.append(values)
                #pdb.set_trace()
                if len(class_values)>1:
                    target = torch.stack(class_values, dim=0)
                elif len(class_values)==1:
                    target = class_values[0].unsqueeze(dim=0)
                target.to(self.device)

                # forward the model
                with torch.set_grad_enabled(False):
                    logits, loss = model(numeric_data, target)    
                    test_loss += loss.item()

                    if isinstance(logits, dict):
                        logit_keys = [x for x in logits.keys() if 'logits' in x]
                        test_correct_inside = []
                        for nk, lk in enumerate(logit_keys):
                            logit = logits[lk]
                            _, predicted = torch.max(logit.data, 1)                            

                            logits_cpu = logit.detach().cpu().numpy()
                            logit_filename = 'val_{}.tsv'.format(lk)
                            if batch_idx_val == 0:
                                f = open(self.complete_save_dir + logit_filename, 'w+')
                                target_handler = self.config.target_handler[nk]
                                header_class = target_handler.classes_
                                write_header = "\t".join(header_class)
                                f.write(write_header)
                                f.write('\ttarget_name\tsample')
                                f.close()
                                
                            f = open(self.complete_save_dir + logit_filename, 'a+')
                            for i_b in range(len(sample_path)):
                                f.write('\n')
                                logits_cpu_flat = logits_cpu[i_b].flatten()
                                logits_cpu_list = logits_cpu_flat.tolist()
                                write_logits = [f'{i:.8f}' for i in logits_cpu_list]
                                target_handler = self.config.target_handler[nk]
                                target_name = target_handler.inverse_transform([target[nk].detach().cpu().numpy().tolist()[i_b]])[0]
                                write_logits.append(str(target_name))
                                write_logits.append(sample_path[i_b])
                                write_header = "\t".join(write_logits)
                                f.write(write_header)
                            f.close()

                            # Ensure target is on the same device as predicted
                            target_on_device = target[nk].to(predicted.device)  # Move target to the same device as predicted
                            test_correct_inside.append(predicted.eq(target_on_device.view_as(predicted)).sum().item())
                        if len(test_correct) == 0:
                            test_correct = np.zeros(len(logit_keys))
                        test_correct += np.asarray(test_correct_inside)
                    else:
                        pass   

            test_loss /= (batch_idx_val+1)
            test_acc = test_correct[0] / len(self.test_dataset) #accuracy based on first target
            print('Validation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                test_loss, test_correct[0], len(self.test_dataset), 100. * test_acc))
            #pdb.set_trace()
            self.save_checkpoint_v3(self.config.save_ckpt_dir)  

            if test_acc > self.global_acc:
                self.global_acc = test_acc
                print(self.global_acc)
                for nk,lk in enumerate(logit_keys):
                    logit_filename = 'val_{}.tsv'.format(lk)
                    shutil.copyfile(self.complete_save_dir + logit_filename, self.complete_save_dir + 'best_' + logit_filename)
                    os.remove(self.complete_save_dir + logit_filename)

                ckpt_path = os.path.join(self.config.save_ckpt_dir, self.config.ckpt_name + '.pthx')
                shutil.copyfile(ckpt_path, self.config.save_ckpt_dir + 'best_ckpt.pthx')

    def unziping_from_package_installation(self):
        pkg_ckpt = resource_filename('muat', 'pkg_ckpt')
        pkg_ckpt = ensure_dirpath(pkg_ckpt)

        all_zip = glob.glob(pkg_ckpt+'*.zip')
        if len(all_zip)>0:
            for checkpoint_file in all_zip:
                with zipfile.ZipFile(checkpoint_file, 'r') as zip_ref:
                    zip_ref.extractall(path=pkg_ckpt)
                os.remove(checkpoint_file) 

    def make_json_serializable(self,obj):
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient="records")  # List of row dicts
        elif isinstance(obj, pd.Series):
            return obj.to_dict()
        elif isinstance(obj, set):
            return list(obj)
        elif hasattr(obj, '__class__') and obj.__class__.__name__ == 'LabelEncoderFromCSV':
            return {
                "class_to_idx": obj.class_to_idx,
                "idx_to_class": obj.idx_to_class,
                "classes_": obj.classes_
            }
        else:
            return obj

    def save_model_config_to_json(self,config, filepath: str):
        def recursive_serialize(obj):
            if isinstance(obj, dict):
                return {k: recursive_serialize(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [recursive_serialize(v) for v in obj]
            else:
                return self.make_json_serializable(obj)

        serialisable_dict = {
            k: recursive_serialize(v)
            for k, v in config.__dict__.items()
        }
        # Set save_ckpt_dir to empty string in the serialized dict
        if 'save_ckpt_dir' in serialisable_dict:
            serialisable_dict['save_ckpt_dir'] = ''
            
        with open(filepath, "w") as f:
            json.dump(serialisable_dict,f)

    def save_dict_to_json(self,data, filepath: str):
        """Helper function to save dictionary data to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(data, f)

    def save_dataframe_to_json(self,df, filepath: str):
        """Helper function to save pandas DataFrame to JSON file."""
        data = df.to_dict(orient="records")
        self.save_dict_to_json(data, filepath)

    def save_checkpoint_v3(self, save_dir: str = None):
        """
        Save the current model state and configuration in v3 format.
        This breaks down the checkpoint into separate files for better organization.
        
        Args:
            save_dir (str, optional): Directory to save the checkpoint. If None, uses config.save_ckpt_path
        """
        if save_dir is None:
            save_dir = self.config.save_ckpt_path
        if save_dir is None:
            raise ValueError("No save directory specified. Either provide save_dir or set config.save_ckpt_path")
            
        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Prepare checkpoint data
        checkpoint = {
            'weight': self.model.state_dict(),
            'target_handler': self.config.target_handler,
            'model_config': self.model.config,
            'trainer_config': self.config,
            'dataloader_config': self.train_dataset.config,
            'model_name': self.model.__class__.__name__,
            'motif_dict': self.model.config.dict_motif,
            'pos_dict': self.model.config.dict_pos,
            'ges_dict': self.model.config.dict_ges
        }
        
        # Save weights
        weights_path = os.path.join(save_dir, 'weight.pth')
        torch.save(checkpoint['weight'], weights_path)

        # Save target handlers
        for idx, handler in enumerate(checkpoint['target_handler']):
            filepath = os.path.join(save_dir, f'target_handler_{idx+1}.json')
            self.save_dict_to_json({
                "class_to_idx": handler.class_to_idx,
                "idx_to_class": handler.idx_to_class,
                "classes_": handler.classes_
            }, filepath)

        # Save configs
        configs = {
            'model_config': checkpoint['model_config'],
            'trainer_config': checkpoint['trainer_config'],
            'dataloader_config': checkpoint['dataloader_config']
        }
        
        for name, config in configs.items():
            filepath = os.path.join(save_dir, f'{name}.json')
            self.save_model_config_to_json(config, filepath)

        # Save model name
        self.save_dict_to_json(checkpoint['model_name'], os.path.join(save_dir, 'model_name.json'))

        # Save dictionaries
        dicts = {
            'motif_dict': checkpoint['motif_dict'],
            'pos_dict': checkpoint['pos_dict'],
            'ges_dict': checkpoint['ges_dict']
        }
        
        for name, df in dicts.items():
            filepath = os.path.join(save_dir, f'{name}.json')
            self.save_dataframe_to_json(df, filepath)

        # Create zip file
        zip_name = self.config.ckpt_name + '.pthx'
        zip_path = os.path.join(save_dir, zip_name)
        
        # Get all .json and .pth files
        files_to_zip = []
        for ext in ['.json', '.pth']:
            files_to_zip.extend(glob.glob(os.path.join(save_dir, f'*{ext}')))
        
        # Create zip file
        
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for file in files_to_zip:
                zipf.write(file, os.path.basename(file))
        
        # Clean up individual files
        for file in files_to_zip:
            os.remove(file)  
        logger.info(f"Checkpoint saved to {zip_path}")

        return zip_path