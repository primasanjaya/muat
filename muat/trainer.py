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

        save_ckpt_params = {'weight':model.module.state_dict(),
                    'target_handler':self.config.target_handler,
                    'model_config':self.model.config,
                    'trainer_config':self.config,
                    'dataloader_config':self.train_dataset.config,
                    'model_name':self.model.__class__.__name__,
                    'motif_dict':self.model.config.dict_motif,
                    'pos_dict':self.model.config.dict_pos,
                    'ges_dict':self.model.config.dict_ges}
            #torch.save(save_ckpt_params, self.config.save_ckpt_dir + 'running_epoch_ckpt_v2.pthx')
        self.convert_checkpoint_v2tov3(self.config.save_ckpt_dir + 'running_epoch_ckpt.pthx', self.config.save_ckpt_dir,save_ckpt_params)

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
            save_ckpt_params = {'weight':model.module.state_dict(),
                    'target_handler':self.config.target_handler,
                    'model_config':self.model.config,
                    'trainer_config':self.config,
                    'dataloader_config':self.train_dataset.config,
                    'model_name':self.model.__class__.__name__,
                    'motif_dict':self.model.config.dict_motif,
                    'pos_dict':self.model.config.dict_pos,
                    'ges_dict':self.model.config.dict_ges}
            #torch.save(save_ckpt_params, self.config.save_ckpt_dir + 'running_epoch_ckpt_v2.pthx')
            convert_checkpoint_v2tov3(self.config.save_ckpt_dir + 'running_epoch_ckpt.pthx', self.config.save_ckpt_dir,save_ckpt_params)

            if test_acc > self.global_acc:
                self.global_acc = test_acc
                print(self.global_acc)
                for nk,lk in enumerate(logit_keys):
                    logit_filename = 'val_{}.tsv'.format(lk)
                    shutil.copyfile(self.complete_save_dir + logit_filename, self.complete_save_dir + 'best_' + logit_filename)
                    os.remove(self.complete_save_dir + logit_filename)
                shutil.copyfile(self.config.save_ckpt_dir + 'running_epoch_ckpt.pthx', self.config.save_ckpt_dir + 'best_ckpt.pthx')


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
        else:
            return obj

    def save_model_config_to_json(self,config, filepath: str):
        serialisable_dict = {
            k: self.make_json_serializable(v)
            for k, v in config.__dict__.items()
        }
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

    def convert_checkpoint_v2tov3(self,ckpt_path, save_dir,checkpoint=None):

        if checkpoint is None:
            # Load the old checkpoint
            checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))
        
        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
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
            'dataloader_config': checkpoint['dataloader_config']
        }
        
        for name, config in configs.items():
            filepath = os.path.join(save_dir, f'{name}.json')
            self.save_model_config_to_json(config, filepath)

        #'trainer_config': checkpoint['trainer_config'],

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

        # Reload all JSON files into a new checkpoint dictionary
        new_checkpoint = {}
        
        # Load weights
        new_checkpoint['weight'] = torch.load(weights_path, map_location=torch.device('cpu'), weights_only=True)

        new_checkpoint['target_handler'] = []
        target_handler_files = [f for f in os.listdir(save_dir) if f.startswith('target_handler_')]
        for file in sorted(target_handler_files):
            filepath = os.path.join(save_dir, file)
            handler = LabelEncoderFromCSV.from_json(filepath)
            new_checkpoint['target_handler'].append(handler)
        
        # Load configs and reconstruct classes
        with open(os.path.join(save_dir, 'model_config.json'), 'r') as f:
            model_config_data = json.load(f)
            # Convert DataFrames back from records
            model_config_data['dict_motif'] = pd.DataFrame(model_config_data['dict_motif'])
            model_config_data['dict_pos'] = pd.DataFrame(model_config_data['dict_pos'])
            model_config_data['dict_ges'] = pd.DataFrame(model_config_data['dict_ges'])
            model_config_data['n_class'] = model_config_data['num_class']
            model_config_data['n_emb'] = model_config_data['n_embd']
            new_checkpoint['model_config'] = ModelConfig(**model_config_data)
        
        with open(os.path.join(save_dir, 'trainer_config.json'), 'r') as f:
            trainer_config_data = json.load(f)
            new_checkpoint['trainer_config'] = TrainerConfig(**trainer_config_data)
        
        with open(os.path.join(save_dir, 'dataloader_config.json'), 'r') as f:
            dataloader_config_data = json.load(f)
            new_checkpoint['dataloader_config'] = DataloaderConfig(**dataloader_config_data)
        
        # Load model name
        with open(os.path.join(save_dir, 'model_name.json'), 'r') as f:
            new_checkpoint['model_name'] = json.load(f)
        
        # Load dictionaries
        for name in ['motif_dict', 'pos_dict', 'ges_dict']:
            with open(os.path.join(save_dir, f'{name}.json'), 'r') as f:
                new_checkpoint[name] = pd.DataFrame(json.load(f))

        # Compare checkpoints
        print("\nComparing checkpoints:")
        for key in checkpoint.keys():
            ck_val = checkpoint[key]
            new_ck_val = new_checkpoint[key]
            
            if key == 'weight':
                # For OrderedDict of tensors, compare each tensor
                if isinstance(ck_val, dict) and isinstance(new_ck_val, dict):
                    is_equal = True
                    for k in ck_val.keys():
                        if k not in new_ck_val:
                            is_equal = False
                            break
                        if not torch.equal(ck_val[k], new_ck_val[k]):
                            is_equal = False
                            break
                else:
                    is_equal = False
                print(f"{key}: {'✓' if is_equal else '✗'}")
            elif key in ['motif_dict', 'pos_dict', 'ges_dict']:
                # For DataFrames, compare values
                is_equal = ck_val.equals(new_ck_val)
                print(f"{key}: {'✓' if is_equal else '✗'}")
            elif key == 'target_handler':
                # For target handlers, compare their attributes
                is_equal = all(
                    h1.class_to_idx == h2.class_to_idx and
                    h1.idx_to_class == h2.idx_to_class and
                    h1.classes_ == h2.classes_
                    for h1, h2 in zip(ck_val, new_ck_val)
                )
                print(f"{key}: {'✓' if is_equal else '✗'}")
            else:
                try:
                    all_ck_keys = ck_val.__dict__.keys()
                    if len(all_ck_keys) > 0:
                        for x in all_ck_keys:
                            cval = vars(ck_val)[x]
                            cnval = vars(new_ck_val)[x]

                            if isinstance(cval, pd.DataFrame):
                                # For DataFrames, use equals() method
                                is_equal = cval.equals(cnval)
                            else:
                                # For other types, use direct comparison
                                is_equal = cval == cnval
                            
                        print(f"{key}: {'✓' if is_equal else '✗'}")
                except:
                    is_equal = True
                    print(f"{key}: {'✓' if is_equal else '✗'}")

        #pdb.set_trace()
        # Create zip file
        zipfile = ckpt_path.split('/')[-1]
        zip_checkpoint_files(save_dir,zipfile)
        
        return new_checkpoint