import torch
import torch.optim as optim
import torch.utils.data
import os
import shutil
import pdb

class TrainerConfig:
    # optimization parameters
    max_epochs = 10
    batch_size = 64
    learning_rate = 3e-4
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.001 # only applied on matmul weights
    # learning rate decay params: linear warmup followed by cosine decay to 10% of original
    lr_decay = False

    # checkpoint settings
    save_ckpt_path = None
    string_logs = None
    num_workers = 0 # for DataLoader
    ckpt_name = 'model'
    args = None

    if save_ckpt_path is not None:
        os.makedirs(ckpt_path, exist_ok=True) 

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

        #string_log = f"{self.config.tag}_{self.config.arch}_fo{self.config.fold:.0f}_bs{self.config.block_size:.0f}_nl{self.config.n_layer:.0f}_nh{self.config.n_head:.0f}_ne{self.config.n_emb:.0f}_ba{self.config.batch_size:.0f}/"
        #self.complete_save_dir = self.config.save_ckpt_dir + string_log
        self.complete_save_dir = self.config.save_ckpt_dir

    def save_checkpoint(self,epoch):
        if self.config.save_ckpt_dir != '':
            #ckpt_model = self.model.module if hasattr(self.model, "module") else self.model
            logger.info("saving %s", ('epoch '+ str(epoch) + self.complete_save_dir + self.config.ckpt_name + '.pthx') )

            ckpt_model = self.model.state_dict()

            ckpt_args = [ckpt_model,self.config.args,self.train_dataset.pd_class_info]
            torch.save(ckpt_args, (self.complete_save_dir + self.config.ckpt_name  + '.pthx'))

    def batch_train(self):
        model, config = self.model, self.config
        model = model.to(self.device)

        model = torch.nn.DataParallel(model).to(self.device)
        optimizer = optim.SGD(model.parameters(), lr=config.learning_rate, momentum=0.9,weight_decay=config.weight_decay)

        batch_size = self.config.batch_size

        trainloader = torch.utils.data.DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        valloader = torch.utils.data.DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False)

        self.global_acc = 0

        for e in range(config.max_epochs):
            running_loss = 0
            model.train(True)

            train_corr = 0
            
            for batch_idx, (data, target) in enumerate(trainloader):
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

                    train_acc = train_corr / len(self.train_dataset)

                    if batch_idx % 100 == 0:
                        print("Epoch {} - Training loss: {:.4f} - Training Acc: {:.2f}".format(e, running_loss/len(self.train_dataset),  train_acc))

            #val
            test_loss = 0
            correct = 0

            self.logit_filename = 'val_logits.tsv'
            f = open(self.complete_save_dir + self.logit_filename, 'w+')  # open file in write mode
            header_class = self.test_dataset.pd_class_info['class_name'].tolist()
            header_class.append('target')
            header_class.append('target_name')
            header_class.append('sample')
            write_header = "\t".join(header_class)
            f.write(write_header)
            f.close()
            
            model.train(False)
            for (data, target) in valloader:
                string_data = data[0]
                numeric_data = data[1]
                for i in range(len(numeric_data)):
                    numeric_data[i] = numeric_data[i].to(self.device)
                target = target.to(self.device)
                # forward the model
                with torch.set_grad_enabled(False):
                    logits, loss = model(numeric_data, target)
                    _, predicted = torch.max(logits.data, 1)

                    predicted = logits.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                    correct += predicted.eq(target.view_as(predicted)).sum().item()

                    #write logits
                    logits_cpu =logits.detach().cpu().numpy()
                    f = open(self.complete_save_dir + self.logit_filename, 'a+')
                    for i in range(numeric_data[0].shape[0]):
                        f.write('\n')
                        logits_cpu_flat = logits_cpu[i].flatten()
                        logits_cpu_list = logits_cpu_flat.tolist()    
                        write_logits = ["%.8f" % i for i in logits_cpu_list]
                        write_logits.append(str(target.detach().cpu().numpy().tolist()[0]))
                        #pdb.set_trace()
                        write_logits.append(string_data[1][i])
                        write_logits.append(string_data[0][i])
                        write_header = "\t".join(write_logits)
                        f.write(write_header)
                    f.close()
                #pdb.set_trace()
            test_loss /= len(self.test_dataset)

            local_acc = correct / len(self.test_dataset)
            print('\Validation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                test_loss, correct, len(self.test_dataset), 100. * local_acc))

            if local_acc > self.global_acc:
                self.global_acc = local_acc
                print(self.global_acc)
                shutil.copyfile(self.complete_save_dir + self.logit_filename, self.complete_save_dir + 'best_vallogits.tsv')
                os.remove(self.complete_save_dir + self.logit_filename)
                self.save_checkpoint(e)
