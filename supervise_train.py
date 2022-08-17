import torch
from torch import nn, optim
import numpy as np
import datetime
import os
import json
from models import *
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

class Runner:
    def __init__(self):
        self.data_dir = './supervise_data'
        self.epochs = 100000
        self.batch_size = 128
        self.learning_rate = 1e-4
    
    def get_file_list(self, path:str, key1:str, key2=''):
        fn_event_file_list = []
        for root, subdirs, filelist in os.walk(path):        
            for f in filelist:
                if key1 in f and key2 in f:
                    fn_event_file_list.append(root + '/' + f)
        return fn_event_file_list
    
    def read_file(self, dir, _type):          
        key = 'json'
        file_list = self.get_file_list(dir, key, 'supervise_data')
        num_samples = len(file_list)
        print('there are {} {} samples'.format(num_samples, _type))
        datas = []  #observation
        labels = []  #constraint_weight
        
        for fname in file_list: 
            with open(fname, 'r') as fid:
                sample = json.load(fid)
                datas.append(sample['observation'])
                labels.append(sample['constraint_weight'])
        return [datas, labels]
    
    def load_data(self):
        train_dir = self.data_dir + '/train'
        valid_dir = self.data_dir + '/valid'
        test_dir = self.data_dir + '/test'
        
        train_sample = self.read_file(train_dir, 'train')
        self.train_data = train_sample[0]
        self.train_label = train_sample[1]
        self.num_train_data = len(self.train_data)
        assert self.num_train_data == len(self.train_label)
        
        valid_sample = self.read_file(valid_dir, 'valid')
        self.valid_data = valid_sample[0]
        self.valid_label = valid_sample[1]    
        self.num_valid_data = len(self.valid_data)
        assert self.num_valid_data == len(self.valid_label)
        
        test_sample = self.read_file(test_dir, 'test')
        self.test_data = test_sample[0]
        self.test_label = test_sample[1]     
        self.num_test_data = len(self.test_data)
        assert self.num_test_data == len(self.test_label)
        
    def load_model(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = my_S2V_QN_scheme3_supervise(self.device).to(self.device)
       
        #para = torch.load('./supervise_checkpoint/model-969627/model-e=10000.pth')
        #self.model.load_state_dict(para)   
        
        self.criterion = torch.nn.MSELoss(reduction = 'sum').to(self.device)
        #self.criterion = torch.nn.CosineSimilarity(1).to(self.device)
        #self.criterion = torch.nn.L1Loss(reduction = 'sum').to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = self.learning_rate)           
        
        self.writer = SummaryWriter('./supervise_log/wright-{}'.format(datetime.datetime.now().microsecond))
        self.writer_train_loss_step = 0    
        self.writer_valid_loss_step = 0   
        self.writer_output_step = 0  
        
        
        self.checkpoint_foldername = './supervise_checkpoint/model-{0}'.format(        
            datetime.datetime.now().microsecond)
        if not os.path.exists(self.checkpoint_foldername):
            os.makedirs(self.checkpoint_foldername) 
        
        return

    def save_model(self, id = 0):
        torch.save(self.model.state_dict(), self.checkpoint_foldername + '/model-e={}.pth'.format(id))
    
    def train_model(self):
        for e in range(self.epochs):
            print('=====epoch {}====='.format(e))
            #Train
            self.model.train()
            random_index = np.random.randint(0, self.num_train_data, size = self.batch_size)
            #print('random_index ',random_index)
            model_output = []
            for i in random_index:
                output_i = self.model(self.train_data[i])
                '''
                for j in range(len(output_i)):
                    self.writer.add_scalars('output' ,{str(j) : output_i[j]}, self.writer_output_step)             
                self.writer_output_step += 1  
                '''
                model_output.append(output_i)
            model_output = torch.stack(model_output).to(self.device)   
            target = []
            for i in random_index:
                target_i = torch.Tensor([self.train_label[i]]).t()
                target.append(target_i)
            target = torch.stack(target).to(self.device)             

            loss = self.criterion(target, model_output)
            

            self.optimizer.zero_grad()
            loss.backward()
            print('train loss = ',loss.item())
            self.optimizer.step()     
            
            self.writer.add_scalars('train_loss' ,{'train_loss':loss},self.writer_train_loss_step)
            self.writer_train_loss_step += 1            
            # Validation loop after each epoch
            self.model.eval()
            with torch.no_grad():
                random_index = np.random.randint(0, self.num_valid_data, size = self.batch_size)
                model_output = []
                for i in random_index:
                    output_i = self.model(self.valid_data[i])
                    model_output.append(output_i)
                model_output = torch.stack(model_output) 
                
                target = []
                for i in random_index:
                    target_i = torch.Tensor([self.valid_label[i]]).t()
                    target.append(target_i)
                target = torch.stack(target).to(self.device)                
            
                loss = self.criterion(target, model_output)
            
                print('valid loss = ',loss.item())
                
                self.writer.add_scalars('valid_loss' ,{'valid_loss' : loss},self.writer_valid_loss_step)
                self.writer_valid_loss_step += 1                      
            if e % 1000 == 0:
                self.save_model(e)
        self.save_model()
    
    def test_model(self):
        print('--------Test---------------')       
        self.model.eval()
        with torch.no_grad():
            
            for i in range(self.num_test_data):
                if i == 1:
                    break                
                print('========test sample{}================='.format(i))
                output_i = self.model(self.test_data[i])#.detach().cpu().numpy()
                output_i = output_i.flatten()
                target_i = self.test_label[i]
                target_i = torch.Tensor(target_i)
                print(output_i)
                print(target_i)

                

            
if __name__ == '__main__':
    R = Runner()
    R.load_data()
    R.load_model()
    R.train_model()
    #R.test_model()