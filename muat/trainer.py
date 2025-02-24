import torch
import torch.optim as optim
import torch.utils.data
import os
import shutil
import pdb
import logging
import numpy as np

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

        if config.save_ckpt_dir is not None:
            os.makedirs(self.config.save_ckpt_dir, exist_ok=True) 

        model = torch.nn.DataParallel(model).to(self.device)
        optimizer = optim.SGD(model.parameters(), lr=self.config.learning_rate, momentum=0.9,weight_decay=config.weight_decay)

        trainloader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.config.batch_size, shuffle=True)
        valloader = torch.utils.data.DataLoader(self.test_dataset, batch_size=self.config.batch_size, shuffle=False)

        self.global_acc = 0

        for e in range(config.max_epochs):
            running_loss = 0
            model.train(True)

            train_corr = 0
            
            for batch_idx, (data, target, sample_path) in enumerate(trainloader):
                string_data = target
                numeric_data = data
                numeric_data = numeric_data.to(self.device)

                target = target.to(self.device)

                # forward the model
                with torch.set_grad_enabled(True):

                    optimizer.zero_grad()
                    logits, loss = model(numeric_data, target)
                    pred = logits.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                    train_corr += pred.eq(target.view_as(pred)).sum().item()
                    
                    loss.backward()
                    #And optimizes its weights here
                    optimizer.step()
                    running_loss += loss.item()
                    #pdb.set_trace()
                    train_acc = train_corr / len(self.train_dataset)

                    if batch_idx % self.config.show_loss_interval == 0:
                        print("Epoch {} - Batch ({}/{}) - Mini-batch Training loss: {:.4f} - Training Acc: {:.2f}".format(e, batch_idx, len(trainloader), running_loss/(batch_idx+1),  train_acc))
            print("Epoch {} - Full-batch Training loss: {:.4f} - Training Acc: {:.2f}".format(e, running_loss/(batch_idx+1),  train_acc))

            #validation
            test_loss = 0
            correct = 0

            self.logit_filename = 'val_logits.tsv'
            #pdb.set_trace()
            f = open(self.complete_save_dir + self.logit_filename, 'w+')  # open file in write mode

            header_class = self.config.target_handler.classes_.tolist()
            header_class.append('target_name')
            header_class.append('sample')
            write_header = "\t".join(header_class)
            f.write(write_header)
            f.close()
            
            model.train(False)
            for batch_idx_val, (data, target, sample_path) in enumerate(valloader):
                string_data = target
                numeric_data = data
                numeric_data = numeric_data.to(self.device)

                target = target.to(self.device)
                # forward the model
                with torch.set_grad_enabled(False):
                    logits, loss = model(numeric_data, target)
                    _, predicted = torch.max(logits.data, 1)

                    test_loss += loss.item()

                    predicted = logits.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                    correct += predicted.eq(target.view_as(predicted)).sum().item()

                    #write logits
                    logits_cpu =logits.detach().cpu().numpy()
                    f = open(self.complete_save_dir + self.logit_filename, 'a+')

                    for i_b in range(len(sample_path)):
                        f.write('\n')
                        logits_cpu_flat = logits_cpu[i_b].flatten()
                        logits_cpu_list = logits_cpu_flat.tolist()
                        write_logits = ["%.8f" % i for i in logits_cpu_list]
                        target_name = self.config.target_handler.inverse_transform([target.detach().cpu().numpy().tolist()[i_b]])[0]
                        write_logits.append(str(target_name))
                        write_logits.append(sample_path[i_b])
                        write_header = "\t".join(write_logits)
                        f.write(write_header)
                    f.close()
                #pdb.set_trace()

            test_loss /= (batch_idx_val+1)
            local_acc = correct / len(self.test_dataset)
            print('Validation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                test_loss, correct, len(self.test_dataset), 100. * local_acc))

            if local_acc > self.global_acc:
                self.global_acc = local_acc
                print(self.global_acc)
                shutil.copyfile(self.complete_save_dir + self.logit_filename, self.complete_save_dir + 'best_vallogits.tsv')
                os.remove(self.complete_save_dir + self.logit_filename)
                
                ckpt_model = self.model.state_dict()

        return ckpt_model

    def batch_train_multilabel(self):
        model = self.model
        model = model.to(self.device)

        if self.config.save_ckpt_dir is not None:
            os.makedirs(self.config.save_ckpt_dir, exist_ok=True) 

        model = torch.nn.DataParallel(model).to(self.device)
        optimizer = optim.SGD(model.parameters(), lr=self.config.learning_rate, momentum=0.9,weight_decay=self.config.weight_decay)

        trainloader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.config.batch_size, shuffle=True)
        valloader = torch.utils.data.DataLoader(self.test_dataset, batch_size=self.config.batch_size, shuffle=False)

        self.global_acc = 0

        for e in range(self.config.max_epochs):
            running_loss = 0
            model.train(True)

            train_corr = []

            num_iterations = 30  # Set limit
            #for batch_idx, (data, target, sample_path) in enumerate(trainloader):
            for batch_idx, (data, target, sample_path) in zip(range(num_iterations), trainloader):

                string_data = target
                numeric_data = data
                numeric_data = numeric_data.to(self.device)

                class_keys = [x for x in string_data.keys() if 'idx_' in x]
                class_values = [string_data[x] for x in class_keys]
                if len(class_values)>1:
                    target = torch.stack(class_values, dim=0)
                else:
                    target = class_values
                target.to(self.device)

                # forward the model
                with torch.set_grad_enabled(True):

                    optimizer.zero_grad()
                    logits, loss = model(numeric_data, target)

                    if isinstance(logits, dict):
                        logit_keys = [x for x in logits.keys() if 'logits' in x]

                        train_corr_inside = []

                        for nk, lk in enumerate(logit_keys):
                            logit = logits[lk]
                            _, pred = torch.max(logit.data, 1)                            
                            logits_cpu =logit.detach().cpu().numpy()

                            pred = logit.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                            train_corr_inside.append(pred.eq(target[nk].view_as(pred)).sum().item())

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
                        show_text = "Epoch {} - Batch ({}/{}) - Mini-batch Training loss: {:.4f}".format(e,batch_idx , len(trainloader) , running_loss/(batch_idx+1))
                        for x in range(len(logit_keys)):
                            show_text = show_text + ' - Training Acc {}: {:.2f}'.format(x+1,train_acc[x])
                        #print(show_text)
            show_text = "Epoch {} - Full-batch Training loss: {:.4f}".format(e, running_loss/(batch_idx+1))
            for x in range(len(logit_keys)):
                show_text = show_text + ' - Training Acc {}: {:.2f}'.format(x+1,train_acc[x])
            print(show_text)

            #validation
            test_loss = 0
            test_correct = []

            if isinstance(logits, dict):
                logit_keys = [x for x in logits.keys() if 'logits' in x]
                for nk,lk in enumerate(logit_keys):
                    logit_filename = 'val_{}.tsv'.format(lk)
                    f = open(self.complete_save_dir + logit_filename, 'w+')  # open file in write mode
                    target_handler = self.config.target_handler[nk]
                    header_class = target_handler.classes_.tolist()
                    header_class.append('target_name')  
                    header_class.append('sample')
                    write_header = "\t".join(header_class)
                    f.write(write_header)
                    f.close()
            
            model.train(False)
            #for batch_idx_val, (data, target, sample_path) in enumerate(valloader):
            for batch_idx_val, (data, target, sample_path) in zip(range(num_iterations), valloader):

                #pdb.set_trace()
                string_data = target
                numeric_data = data
                numeric_data = numeric_data.to(self.device)

                class_keys = [x for x in string_data.keys() if 'idx_' in x]
                class_values = [string_data[x] for x in class_keys]
                if len(class_values)>1:
                    target = torch.stack(class_values, dim=0)
                else:
                    target = class_values
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

                            logits_cpu =logit.detach().cpu().numpy()
                            logit_filename = 'val_{}.tsv'.format(lk)
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
                            test_correct_inside.append(predicted.eq(target[nk].view_as(predicted)).sum().item())
                        if len(test_correct) == 0:
                            test_correct = np.zeros(len(logit_keys))
                        test_correct += np.asarray(test_correct_inside)
                    else:
                        pass   

            test_loss /= (batch_idx_val+1)
            test_acc = test_correct[0] / len(self.test_dataset) #accuracy based on first target
            print('Validation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                test_loss, test_correct[0], len(self.test_dataset), 100. * test_acc))

            if test_acc > self.global_acc:
                self.global_acc = test_acc
                print(self.global_acc)
                for nk,lk in enumerate(logit_keys):
                    logit_filename = 'val_{}.tsv'.format(lk)
                    shutil.copyfile(self.complete_save_dir + logit_filename, self.complete_save_dir + 'best_' + logit_filename)
                    os.remove(self.complete_save_dir + logit_filename)
                
                ckpt_model = self.model.state_dict()

        return ckpt_model