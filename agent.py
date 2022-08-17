import numpy as np
import random
import time
import os
from models import *
import copy
import datetime

import torch.nn.functional as F
import torch
import torch.nn as nn

from torch.utils.tensorboard import SummaryWriter


class DQAgent:
    def __init__(self, problem_list, lr, bs, n_step, pattern):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.problem_list = problem_list
        self.pattern = pattern
        
        self.k = 20
        self.alpha = 0.1
        self.gamma = 0.99
        self.lambd = 0.
        self.n_step = n_step

        self.epsilon_ = 1
        self.epsilon_min = 0.02
        self.discount_factor = 0.99999#################################################

        self.t = 1
        self.memory = []
        self.memory_n = []
        self.minibatch_length = bs

        self.load_model()
        self.criterion = torch.nn.MSELoss(reduction = 'sum')
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = lr)
        
        self.delay_remember = []
    
    def load_model(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('[device]',device)
        self.model = my_S2V_QN_scheme3(device).to(device)
        if self.pattern == 'eval':
            print('[pattern] eval')
            para = torch.load('./checkpoint/xxx.pth')
            self.model.load_state_dict(para)            
            self.epsilon_ = -1 #############################不随机选择
        elif self.pattern == 'train':
            print('[pattern] train')
            #para = torch.load('./checkpoint/xxx.pth')
            #self.model.load_state_dict(para)   
                  
            self.writer = SummaryWriter('./log/wright-{}'.format(datetime.datetime.now().microsecond))
            self.writer_loss_step = 0       
            self.writer_cum_reward_step = 0     
            self.writer_action_step = 0   
        
    
    def save_model(self):
        torch.save(self.model.state_dict(), './checkpoint/xxx.pth')    #new-model_test_0.5

    def wright(self, cum_reward):
        self.writer.add_scalars('cum_reward' ,{'cum_reward':cum_reward},self.writer_cum_reward_step)
        self.writer_cum_reward_step += 1        
    
    def reset(self, problem_id):
        self.problem_id = problem_id
        self.problem_list[self.problem_id].re_initialize_num_server()
        max_delay, user_i, user_j = self.problem_list[self.problem_id].get_max_queuing_delay_in_interaction_delay()
        cost = self.problem_list[self.problem_id].compute_cost()
        print('=======reset cost={}, budget={} [max delay]={} '.format(cost, self.problem_list[self.problem_id]._cost_budget, round(max_delay*1000, 2)))        
        
        '''
        if (len(self.memory_n) != 0) and (len(self.memory_n) % 30000 == 0):  #300000
            self.memory_n = random.sample(self.memory_n, 10000)  #120000
        
        if len(self.memory) > 1000:
            self.memory = self.memory[-500:]
        '''

        self.last_action = 0    
        self.last_observation = self.problem_list[self.problem_id].observe()
        
        self.last_reward = -0.01
        self.last_done = False
        self.iter = 1
        self.delay_remember = []

    def act(self, observation):
        valid_action_mask = self.problem_list[self.problem_id].valid_action_mask()
        if self.epsilon_ > np.random.rand():
            valid_index = []
            for i in range(len(valid_action_mask)):
                if valid_action_mask[i]:
                    valid_index.append(i)
            action = np.random.choice(valid_index)            
            #print('random action]: ',action)            
            return action
        else:
            q_a = self.model(observation)
            q_a = q_a.detach().cpu().numpy()
            q_a = q_a.flatten()
            if self.pattern == 'train':
                for j in range(len(q_a)):
                    self.writer.add_scalars('action' ,{str(j) : q_a[j]}, self.writer_action_step)            
                self.writer_action_step += 1
            
            valid_index = []
            valid_Q = []
            for i in range(len(valid_action_mask)):
                if valid_action_mask[i]:
                    valid_index.append(i)
                    valid_Q.append(q_a[i])
            max_index = np.argmax(valid_Q)

            action = valid_index[max_index]
            #print('[Q output] ',q_a)
            print(action, '   ', end=' ')
            return action  

    def reward(self, observation, action, reward, done):
        if len(self.memory_n) > self.minibatch_length + self.n_step: #or self.games > 2:
            (last_observation_tens, action_tens, reward_tens, observation_tens, done_tens) = self.get_sample()
            obs_model_output = []
            for i in range(len(observation_tens)):
                output_i = self.model(observation_tens[i])
                obs_model_output.append(output_i)
            obs_model_output = torch.stack(obs_model_output) 
            #target = reward_tens + self.gamma * (1-done_tens) * torch.max(self.model(observation_tens), dim=1)[0] #+ observation_tens * (-1e5), dim=1)[0]
            target = reward_tens + self.gamma * (1-done_tens) * torch.max(obs_model_output, dim=1)[0]
            last_obs_model_output = []
            for i in range(len(last_observation_tens)):
                last_obs_model_output.append(self.model(last_observation_tens[i]))            
            target_f = torch.stack(last_obs_model_output)
            target_p = target_f.clone()
            target_f[range(self.minibatch_length), action_tens, : ] = target

            loss = self.criterion(target_p, target_f)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            self.writer.add_scalars('loss' ,{'loss':loss},self.writer_loss_step)
            self.writer_loss_step += 1
            
            #self.epsilon = self.eps_end + max(0., (self.eps_start- self.eps_end) * (self.eps_step - self.t) / self.eps_step)
            if self.epsilon_ > self.epsilon_min:
                self.epsilon_ *= self.discount_factor
        
        if self.iter>1:
            self.remember(self.last_observation, self.last_action, self.last_reward, observation, self.last_done * 1 )

        if done and self.iter> self.n_step:
            self.remember_n(False)
            new_observation =  self.problem_list[self.problem_id].observe()
            self.remember(observation, action, reward, new_observation, done*1)

        if self.iter > self.n_step:
            self.remember_n(done)
            
        self.iter += 1
        self.last_action = action
        self.last_observation = observation
        self.last_reward = reward
        self.last_done = done

    def get_sample(self):
        #obs == [delay_feature, service_feature, edge_feature, edge_indice, adj]
        minibatch = random.sample(self.memory_n, self.minibatch_length - 1)
        minibatch.append(self.memory_n[-1])
        last_observation_tens = [minibatch[0][0]]
        action_tens = torch.Tensor([minibatch[0][1]]).type(torch.LongTensor).to(self.device)
        reward_tens = torch.Tensor([[minibatch[0][2]]]).to(self.device)
        observation_tens = [minibatch[0][3]]
        done_tens =torch.Tensor([[minibatch[0][4]]]).to(self.device)

        for last_observation_, action_, reward_, observation_, done_, games_ in minibatch[-self.minibatch_length + 1:]:

            last_observation_tens.append(last_observation_)
            #last_observation_tens = torch.cat((last_observation_tens,last_observation_))
            action_tens = torch.cat((action_tens, torch.Tensor([action_]).type(torch.LongTensor).to(self.device)))
            reward_tens = torch.cat((reward_tens, torch.Tensor([[reward_]]).to(self.device)))
            observation_tens.append(observation_)
            #observation_tens = torch.cat((observation_tens, observation_))      
            done_tens = torch.cat((done_tens, torch.Tensor([[done_]]).to(self.device)))
        
        return (last_observation_tens, action_tens, reward_tens, observation_tens, done_tens)



    def remember(self, last_observation, last_action, last_reward, observation, done):
        self.memory.append((last_observation, last_action, last_reward, observation, done, self.problem_id))

    def remember_n(self, done):
        if not done:
            step_init = self.memory[-self.n_step]
            cum_reward = step_init[2]
            for step in range(1, self.n_step):
                cum_reward += self.memory[-step][2]
            self.memory_n.append((step_init[0], step_init[1], cum_reward, self.memory[-1][-3], self.memory[-1][-2], self.memory[-1][-1]))

        else:
            for i in range(1, self.n_step):
                step_init = self.memory[-self.n_step + i]
                cum_reward = step_init[2]
                for step in range(1, self.n_step - i):
                    cum_reward += self.memory[-step][2]
                if i == self.n_step - 1:
                    self.memory_n.append(
                        (step_init[0], step_init[1], cum_reward, self.memory[-1][-3], False, self.memory[-1][-1]))
                else:
                    self.memory_n.append((step_init[0], step_init[1], cum_reward, self.memory[-1][-3], False, self.memory[-1][-1]))

