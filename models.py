import torch
import torch.nn.functional as F
import torch.nn as nn
import random
import numpy as np

#=======================original============================================
class S2V_QN_1(torch.nn.Module):
    def __init__(self,reg_hidden, embed_dim, len_pre_pooling, len_post_pooling, T):

        super(S2V_QN_1, self).__init__()
        self.T = T
        self.embed_dim=embed_dim
        self.reg_hidden=reg_hidden
        self.len_pre_pooling = len_pre_pooling
        self.len_post_pooling = len_post_pooling
        #self.mu_1 = torch.nn.Linear(1, embed_dim)
        #torch.nn.init.normal_(self.mu_1.weight,mean=0,std=0.01)
        self.mu_1 = torch.nn.Parameter(torch.Tensor(1, embed_dim))
        torch.nn.init.normal_(self.mu_1, mean=0, std=0.01)
        self.mu_2 = torch.nn.Linear(embed_dim, embed_dim,True)
        torch.nn.init.normal_(self.mu_2.weight, mean=0, std=0.01)
        self.list_pre_pooling = []
        for i in range(self.len_pre_pooling):
            pre_lin = torch.nn.Linear(embed_dim,embed_dim,bias=True)
            torch.nn.init.normal_(pre_lin.weight, mean=0, std=0.01)
            self.list_pre_pooling.append(pre_lin)
        self.list_post_pooling = []
        for i in range(self.len_post_pooling):
            post_lin =torch.nn.Linear(embed_dim,embed_dim,bias=True)
            torch.nn.init.normal_(post_lin.weight, mean=0, std=0.01)
            self.list_post_pooling.append(post_lin)
        self.q_1 = torch.nn.Linear(embed_dim, embed_dim,bias=True)
        torch.nn.init.normal_(self.q_1.weight, mean=0, std=0.01)
        self.q_2 = torch.nn.Linear(embed_dim, embed_dim,bias=True)
        torch.nn.init.normal_(self.q_2.weight, mean=0, std=0.01)
        if self.reg_hidden > 0:
            self.q_reg = torch.nn.Linear(2 * embed_dim, self.reg_hidden)
            torch.nn.init.normal_(self.q_reg.weight, mean=0, std=0.01)
            self.q = torch.nn.Linear(self.reg_hidden, 1)
        else:
            self.q = torch.nn.Linear(2 * embed_dim, 1)
        torch.nn.init.normal_(self.q.weight, mean=0, std=0.01)

    def forward(self, xv, adj):

        minibatch_size = xv.shape[0]
        nbr_node = xv.shape[1]


        for t in range(self.T):
            if t == 0:
                #mu = self.mu_1(xv).clamp(0)
                mu = torch.matmul(xv, self.mu_1).clamp(0)
                #mu.transpose_(1,2)
                #mu_2 = self.mu_2(torch.matmul(adj, mu_init))
                #mu = torch.add(mu_1, mu_2).clamp(0)

            else:
                #mu_1 = self.mu_1(xv).clamp(0)
                mu_1 = torch.matmul(xv, self.mu_1).clamp(0)
                #mu_1.transpose_(1,2)
                # before pooling:
                for i in range(self.len_pre_pooling):
                    mu = self.list_pre_pooling[i](mu).clamp(0)

                mu_pool = torch.matmul(adj, mu)

                # after pooling
                for i in range(self.len_post_pooling):
                    mu_pool = self.list_post_pooling[i](mu_pool).clamp(0)

                mu_2 = self.mu_2(mu_pool)
                mu = torch.add(mu_1, mu_2).clamp(0)

        q_1 = self.q_1(torch.matmul(xv.transpose(1,2),mu)).expand(minibatch_size,nbr_node,self.embed_dim)
        q_2 = self.q_2(mu)
        q_ = torch.cat((q_1, q_2), dim=-1)
        if self.reg_hidden > 0:
            q_reg = self.q_reg(q_).clamp(0)
            q = self.q(q_reg)
        else:
            q_=q_.clamp(0)
            q = self.q(q_)
        return q
#=================original==================================================

class EmbeddingModule(nn.Module):
    def __init__(self, n_feats, emb_size):
        super().__init__()
        self.layer = nn.Sequential()
        l1 = nn.BatchNorm1d(n_feats)
        l2 = nn.Linear(n_feats, emb_size)
        #torch.nn.init.normal_(l2.weight, mean=0, std=0.01)
        l3 = nn.ReLU()
        l4 = nn.Linear(emb_size, emb_size)
        #torch.nn.init.normal_(l4.weight, mean=0, std=0.01)
        l5 = nn.ReLU()
        self.layer.add_module('l1', l1)
        self.layer.add_module('l2', l2)
        self.layer.add_module('l3', l3)
        self.layer.add_module('l4', l4)
        self.layer.add_module('l5', l5)


    def forward(self, input):
        return self.layer(input)

class EdgeEmbeddingModule(nn.Module):
    """
    This class will only be used for edge embedding
    """
    def __init__(self, n_feats):
        super().__init__()
        self.pre_norm_layer = nn.BatchNorm1d(n_feats)

    def forward(self, input):
        return self.pre_norm_layer(input)

class BipartiteGraphConvolution(nn.Module):
    """
    Partial bipartite graph convolution (either left-to-right or right-to-left).
    """

    def __init__(self, emb_size, right_to_left=False, device=None):
        super().__init__()
        self.device = device
        self.emb_size = emb_size
        self.right_to_left = right_to_left

        self.feature_module_left = nn.Linear(self.emb_size, self.emb_size).to(self.device)
        nn.init.orthogonal_(self.feature_module_left.weight)

        self.feature_module_edge = nn.Linear(1, self.emb_size, bias=False).to(self.device)
        nn.init.orthogonal_(self.feature_module_edge.weight)

        self.feature_module_right = nn.Linear(self.emb_size, self.emb_size, bias=False).to(self.device)
        nn.init.orthogonal_(self.feature_module_right.weight)

        self.feature_model_final = nn.Sequential(
            nn.BatchNorm1d(self.emb_size),
            nn.ReLU(),
            nn.Linear(self.emb_size, self.emb_size)
        ).to(self.device)

        self.post_conv_module = nn.BatchNorm1d(self.emb_size).to(self.device)

        self.output_module = nn.Sequential(
            nn.Linear(2 * self.emb_size, self.emb_size),
            nn.ReLU(),
            nn.Linear(self.emb_size, self.emb_size),
            nn.ReLU()
        ).to(self.device)

    def forward(self, inputs):
        """
        Performs a partial graph convolution on the given bipartite graph.
        Inputs
        ------
        left_features: 2D float tensor
            Features of the left-hand-side nodes in the bipartite graph
        edge_indices: 2D int tensor
            Edge indices in left-right order
        edge_features: 2D float tensor
            Features of the edges
        right_features: 2D float tensor
            Features of the right-hand-side nodes in the bipartite graph
        scatter_out_size: 1D int tensor
            Output size (left_features.shape[0] or right_features.shape[0], unknown at compile time)
        """
        left_features, edge_indices, edge_features, right_features, scatter_out_size = inputs

        if self.right_to_left:
            scatter_dim = 0
            prev_features = left_features
        else:
            scatter_dim = 1
            prev_features = right_features

        joint_features = self.feature_module_edge(edge_features)
        joint_features.add_(self.feature_module_right(right_features)[edge_indices[1]])
        joint_features.add_(self.feature_module_left(left_features)[edge_indices[0]])
        
        joint_features = self.feature_model_final(joint_features)

        conv_output = torch.zeros([scatter_out_size, self.emb_size])\
            .to(self.device).index_add(0, edge_indices[scatter_dim], joint_features)

        conv_output = self.post_conv_module(conv_output)

        output = torch.cat((conv_output, prev_features), dim=1)
        output = self.output_module(output)
        return output




class my_S2V_QN_scheme2(nn.Module):
    def __init__(self, device):

        super(my_S2V_QN_scheme2, self).__init__()
        self.device = device
        self.emb_size = 64
        self.delay_nfeats = 1
        self.edge_nfeats = 1
        self.service_nfeats = 6
    
        self.delay_embedding = EmbeddingModule(self.delay_nfeats, self.emb_size).to(device)
        self.edge_embedding = EdgeEmbeddingModule(self.edge_nfeats).to(device)
        self.service_embedding = EmbeddingModule(self.service_nfeats, self.emb_size).to(device)        
        
        self.T = 1

        self.theta_1 = nn.Linear(self.emb_size, self.emb_size, True).to(device)
        nn.init.normal_(self.theta_1.weight, mean=0, std=0.01)

        self.theta_2 = nn.Linear(self.emb_size, self.emb_size, True).to(device)
        nn.init.normal_(self.theta_2.weight, mean=0, std=0.01)
        
        self.theta_3 = nn.Linear(self.emb_size, self.emb_size, True).to(device)    
        nn.init.normal_(self.theta_3.weight, mean=0, std=0.01)        
        
        self.theta_4 = nn.Sequential().to(device) 
        theta_4_1 = nn.Linear(1, self.emb_size, True).to(device)
        nn.init.normal_(theta_4_1.weight, mean=0, std=0.01)
        theta_4_2 = nn.ReLU().to(device)        
        self.theta_4.add_module('linear', theta_4_1)
        self.theta_4.add_module('relu', theta_4_2)
        
        self.sum_theta_ReLU = nn.ReLU().to(device)      
        
        self.theta_6 = nn.Linear(self.emb_size, self.emb_size, True).to(device)    
        nn.init.normal_(self.theta_6.weight, mean=0, std=0.01)        
        
        self.theta_7 = nn.Linear(self.emb_size, self.emb_size, True).to(device)    
        nn.init.normal_(self.theta_7.weight, mean=0, std=0.01)           
        
        self.theta_5 = nn.Linear(2 * self.emb_size, 1).to(device)
        nn.init.normal_(self.theta_5.weight, mean=0, std=0.01)  
        
        self.cat_67_ReLU = nn.ReLU().to(device)     

    def forward(self, obs): #obs
        #print('------forward-------------')
        #obs == [delay_feature, service_feature, edge_feature, edge_indice, adj]
        
        delay_feature = torch.Tensor(obs[0]).to(self.device)
        service_feature = torch.Tensor(obs[1]).to(self.device)
        adj = torch.Tensor(obs[4]).to(self.device)
        delay_ones = torch.ones(len(delay_feature),1).to(self.device)
        service_ones = torch.ones(1, len(service_feature)).to(self.device)

        #print('========input ')
        #print('delay_feature ',delay_feature.shape)
        #print('service_feature ' ,service_feature.shape)
        #print('adj ',adj.shape)
        
        service_feature = self.service_embedding(service_feature)
        delay_feature = self.delay_embedding(delay_feature)
        

        for i in range(self.T):
            theta_1 = self.theta_1(service_feature)

            sum_neighbors_delay = torch.matmul(adj, delay_feature)
            theta_2 = self.theta_2(sum_neighbors_delay)
            
            weight_one = torch.Tensor([[1]]).to(self.device)
            theta_4 = self.theta_4(weight_one)
            sum_edge_weight = torch.matmul(adj, delay_ones)
            sum_edge_weight = torch.matmul(sum_edge_weight, theta_4)  
            
            theta_3 = self.theta_3(sum_edge_weight)     
            
            sum_theta = torch.add(theta_1, theta_2)
            sum_theta = torch.add(sum_theta, theta_3)
            sum_theta = self.sum_theta_ReLU(sum_theta)
            service_feature = sum_theta.clone()

        sum_service_feature = torch.matmul(service_ones, service_feature).expand(len(service_feature), self.emb_size)
        theta_6 = self.theta_6(sum_service_feature)
        theta_7 = self.theta_7(service_feature)
        c67 = torch.cat((theta_6, theta_7), dim=-1)
        c67 = self.cat_67_ReLU(c67)
        
        Q = self.theta_5(c67)
        return Q  
    
class my_S2V_QN_scheme3_supervise(nn.Module):
    def __init__(self, device):

        super(my_S2V_QN_scheme3_supervise, self).__init__()
        self.device = device
        self.emb_size = 64
        self.delay_nfeats = 1
        self.edge_nfeats = 1
        self.service_nfeats = 6
    
        self.delay_embedding = EmbeddingModule(self.delay_nfeats, self.emb_size).to(device)
        self.edge_embedding = EdgeEmbeddingModule(self.edge_nfeats).to(device)
        self.service_embedding = EmbeddingModule(self.service_nfeats, self.emb_size).to(device)        
        '''
        self.theta_1 = nn.Linear(self.emb_size, self.emb_size, True).to(device)
        nn.init.normal_(self.theta_1.weight, mean=0, std=0.01)

        self.theta_2 = nn.Linear(self.emb_size, self.emb_size, True).to(device)
        nn.init.normal_(self.theta_2.weight, mean=0, std=0.01)
        
        self.theta_3 = nn.Linear(self.emb_size, self.emb_size, True).to(device)    
        nn.init.normal_(self.theta_3.weight, mean=0, std=0.01)        
        '''
        ########delay
        self.theta_1_delay = nn.Linear(self.emb_size, self.emb_size, True).to(device)
        nn.init.normal_(self.theta_1_delay.weight, mean=0, std=0.01)
    
        self.theta_2_delay = nn.Linear(self.emb_size, self.emb_size, True).to(device)
        nn.init.normal_(self.theta_2_delay.weight, mean=0, std=0.01)
    
        self.theta_3_delay = nn.Linear(self.emb_size, self.emb_size, True).to(device)    
        nn.init.normal_(self.theta_3_delay.weight, mean=0, std=0.01)      
    
        ########service
        self.theta_1_service = nn.Linear(self.emb_size, self.emb_size, True).to(device)
        nn.init.normal_(self.theta_1_service.weight, mean=0, std=0.01)
    
        self.theta_2_service = nn.Linear(self.emb_size, self.emb_size, True).to(device)
        nn.init.normal_(self.theta_2_service.weight, mean=0, std=0.01)
    
        self.theta_3_service = nn.Linear(self.emb_size, self.emb_size, True).to(device)    
        nn.init.normal_(self.theta_3_service.weight, mean=0, std=0.01)            
        ####################
        self.theta_4 = nn.Sequential().to(device) 
        theta_4_1 = nn.Linear(1, self.emb_size, True).to(device)
        nn.init.normal_(theta_4_1.weight, mean=0, std=0.01)
        theta_4_2 = nn.ReLU().to(device)        
        self.theta_4.add_module('linear', theta_4_1)
        self.theta_4.add_module('relu', theta_4_2)
        
        self.sum_theta_ReLU = nn.ReLU().to(device)      
        
        self.theta_6 = nn.Linear(self.emb_size, self.emb_size, True).to(device)    
        nn.init.normal_(self.theta_6.weight, mean=0, std=0.01)        
        
        self.theta_7 = nn.Linear(self.emb_size, self.emb_size, True).to(device)    
        nn.init.normal_(self.theta_7.weight, mean=0, std=0.01)           
        
        self.theta_5 = nn.Linear(2 * self.emb_size, 1).to(device)
        nn.init.normal_(self.theta_5.weight, mean=0, std=0.01)  
        
        self.cat_67_ReLU = nn.ReLU().to(device) 
        
        self.output_Softmax = nn.Softmax(dim=0)
    
    def topk_softmax(self, Q):

        top_value, top_index = torch.topk(Q, 5, dim=0)   #[0]value   [1]index

        softmax_Q_topk = self.output_Softmax(top_value)

        Q_zero = torch.zeros_like(Q)

        Q_zero[top_index.flatten().type(torch.LongTensor)] = softmax_Q_topk
        
        return Q_zero
    
    def get_delay_feature(self, service_feature, delay_feature, adj):
        adj_t = adj.t()
        service_ones = torch.ones(len(service_feature), 1).to(self.device)        
        theta_1 = self.theta_1_delay(delay_feature)
    
        sum_neighbors_delay = torch.matmul(adj_t, service_feature)
        theta_2 = self.theta_2_delay(sum_neighbors_delay)
    
        weight_one = torch.Tensor([[1]]).to(self.device)
        theta_4 = self.theta_4(weight_one)
        sum_edge_weight = torch.matmul(adj_t, service_ones)
        sum_edge_weight = torch.matmul(sum_edge_weight, theta_4)  
    
        theta_3 = self.theta_3_delay(sum_edge_weight)     
    
        sum_theta = torch.add(theta_1, theta_2)
        sum_theta = torch.add(sum_theta, theta_3)
        sum_theta = self.sum_theta_ReLU(sum_theta)  
        return sum_theta
    
    def get_service_feature(self, service_feature, delay_feature, adj):
        delay_ones = torch.ones(len(delay_feature),1).to(self.device)
        theta_1 = self.theta_1_service(service_feature)
    
        sum_neighbors_delay = torch.matmul(adj, delay_feature)
        theta_2 = self.theta_2_service(sum_neighbors_delay)
    
        weight_one = torch.Tensor([[1]]).to(self.device)
        theta_4 = self.theta_4(weight_one)
        sum_edge_weight = torch.matmul(adj, delay_ones)
        sum_edge_weight = torch.matmul(sum_edge_weight, theta_4)  
    
        theta_3 = self.theta_3_service(sum_edge_weight)     
    
        sum_theta = torch.add(theta_1, theta_2)
        sum_theta = torch.add(sum_theta, theta_3)
        sum_theta = self.sum_theta_ReLU(sum_theta)
        return sum_theta    
    

    def forward(self, obs): #obs
        #print('------forward-------------')
        #obs == [delay_feature, service_feature, edge_feature, edge_indice, adj]
        
        delay_feature = torch.Tensor(obs[0]).to(self.device)
        service_feature = torch.Tensor(obs[1]).to(self.device)
        adj = torch.Tensor(obs[4]).to(self.device)
        adj_t = adj.t()

        #print('========input ')
        #print('delay_feature ',delay_feature.shape)
        #print('service_feature ' ,service_feature.shape)
        #print('adj ',adj.shape)
        
        service_feature = self.service_embedding(service_feature)
        delay_feature = self.delay_embedding(delay_feature)
        ####################################################
        service_feature = self.get_service_feature(service_feature, delay_feature, adj)  
        delay_feature = self.get_delay_feature(service_feature, delay_feature, adj)
              
        
        ###################################################
        delay_ones = torch.ones(1, len(delay_feature)).to(self.device)
        sum_delay_feature = torch.matmul(delay_ones, delay_feature).expand(len(delay_feature), self.emb_size)
        theta_6 = self.theta_6(sum_delay_feature)
        theta_7 = self.theta_7(delay_feature)
        c67 = torch.cat((theta_6, theta_7), dim=-1)
        c67 = self.cat_67_ReLU(c67)
        
        Q = self.theta_5(c67)
        #Q = self.topk_softmax(Q)
        Q = self.output_Softmax(Q)
        return Q   
    
class my_S2V_QN_scheme3_learn_delay_weight(nn.Module):
    def __init__(self, device):

        super(my_S2V_QN_scheme3_learn_delay_weight, self).__init__()
        self.device = device
        self.emb_size = 64
        self.delay_nfeats = 1
        self.edge_nfeats = 1
        self.service_nfeats = 5
    
        self.delay_embedding = EmbeddingModule(self.delay_nfeats, self.emb_size).to(device)
        self.edge_embedding = EdgeEmbeddingModule(self.edge_nfeats).to(device)
        self.service_embedding = EmbeddingModule(self.service_nfeats, self.emb_size).to(device)        

        ########delay
        self.theta_1_delay = nn.Linear(self.emb_size, self.emb_size, True).to(device)
        nn.init.normal_(self.theta_1_delay.weight, mean=0, std=0.01)
    
        self.theta_2_delay = nn.Linear(self.emb_size, self.emb_size, True).to(device)
        nn.init.normal_(self.theta_2_delay.weight, mean=0, std=0.01)
    
        self.theta_3_delay = nn.Linear(self.emb_size, self.emb_size, True).to(device)    
        nn.init.normal_(self.theta_3_delay.weight, mean=0, std=0.01)      
    
        ########service
        self.theta_1_service = nn.Linear(self.emb_size, self.emb_size, True).to(device)
        nn.init.normal_(self.theta_1_service.weight, mean=0, std=0.01)
    
        self.theta_2_service = nn.Linear(self.emb_size, self.emb_size, True).to(device)
        nn.init.normal_(self.theta_2_service.weight, mean=0, std=0.01)
    
        self.theta_3_service = nn.Linear(self.emb_size, self.emb_size, True).to(device)    
        nn.init.normal_(self.theta_3_service.weight, mean=0, std=0.01)            
        ####################
        self.theta_4 = nn.Sequential().to(device) 
        theta_4_1 = nn.Linear(1, self.emb_size, True).to(device)
        nn.init.normal_(theta_4_1.weight, mean=0, std=0.01)
        theta_4_2 = nn.ReLU().to(device)        
        self.theta_4.add_module('linear', theta_4_1)
        self.theta_4.add_module('relu', theta_4_2)
        
        self.sum_theta_ReLU = nn.ReLU().to(device)      
        
        self.theta_6 = nn.Linear(self.emb_size, self.emb_size, True).to(device)    
        nn.init.normal_(self.theta_6.weight, mean=0, std=0.01)        
        
        self.theta_7 = nn.Linear(self.emb_size, self.emb_size, True).to(device)    
        nn.init.normal_(self.theta_7.weight, mean=0, std=0.01)           
        
        self.theta_5 = nn.Linear(2 * self.emb_size, 1).to(device)
        nn.init.normal_(self.theta_5.weight, mean=0, std=0.01)  
        
        self.cat_67_ReLU = nn.ReLU().to(device) 
        
        #self.output_Softmax = nn.Softmax(dim=0)
    
    def get_delay_feature(self, service_feature, delay_feature, adj):
        adj_t = adj.t()
        service_ones = torch.ones(len(service_feature), 1).to(self.device)        
        theta_1 = self.theta_1_delay(delay_feature)
    
        sum_neighbors_delay = torch.matmul(adj_t, service_feature)
        theta_2 = self.theta_2_delay(sum_neighbors_delay)
    
        weight_one = torch.Tensor([[1]]).to(self.device)
        theta_4 = self.theta_4(weight_one)
        sum_edge_weight = torch.matmul(adj_t, service_ones)
        sum_edge_weight = torch.matmul(sum_edge_weight, theta_4)  
    
        theta_3 = self.theta_3_delay(sum_edge_weight)     
    
        sum_theta = torch.add(theta_1, theta_2)
        sum_theta = torch.add(sum_theta, theta_3)
        sum_theta = self.sum_theta_ReLU(sum_theta)  
        return sum_theta
    
    def get_service_feature(self, service_feature, delay_feature, adj):
        delay_ones = torch.ones(len(delay_feature),1).to(self.device)
        theta_1 = self.theta_1_service(service_feature)
    
        sum_neighbors_delay = torch.matmul(adj, delay_feature)
        theta_2 = self.theta_2_service(sum_neighbors_delay)
    
        weight_one = torch.Tensor([[1]]).to(self.device)
        theta_4 = self.theta_4(weight_one)
        sum_edge_weight = torch.matmul(adj, delay_ones)
        sum_edge_weight = torch.matmul(sum_edge_weight, theta_4)  
    
        theta_3 = self.theta_3_service(sum_edge_weight)     
    
        sum_theta = torch.add(theta_1, theta_2)
        sum_theta = torch.add(sum_theta, theta_3)
        sum_theta = self.sum_theta_ReLU(sum_theta)
        return sum_theta    
    

    def forward(self, obs): #obs
        #print('------forward-------------')
        #obs == [delay_feature, service_feature, edge_feature, edge_indice, adj]
        
        delay_feature = torch.Tensor(obs[0]).to(self.device)
        service_feature = torch.Tensor(obs[1]).to(self.device)
        adj = torch.Tensor(obs[4]).to(self.device)
        adj_t = adj.t()

        #print('========input ')
        #print('delay_feature ',delay_feature.shape)
        #print('service_feature ' ,service_feature.shape)
        #print('adj ',adj.shape)
        
        service_feature = self.service_embedding(service_feature)
        delay_feature = self.delay_embedding(delay_feature)
        ####################################################
        service_feature = self.get_service_feature(service_feature, delay_feature, adj)  
        delay_feature = self.get_delay_feature(service_feature, delay_feature, adj)
              
        
        ###################################################
        delay_ones = torch.ones(1, len(delay_feature)).to(self.device)
        sum_delay_feature = torch.matmul(delay_ones, delay_feature).expand(len(delay_feature), self.emb_size)
        theta_6 = self.theta_6(sum_delay_feature)
        theta_7 = self.theta_7(delay_feature)
        c67 = torch.cat((theta_6, theta_7), dim=-1)
        c67 = self.cat_67_ReLU(c67)
        
        Q = self.theta_5(c67)
        return Q       
    
    
'''
class my_S2V_QN_scheme1(nn.Module):
    def __init__(self, device):

        super(my_S2V_QN_scheme1, self).__init__()
        self.device = device
        self.emb_size = 64
        self.delay_nfeats = 1
        self.edge_nfeats = 1
        self.service_nfeats = 6
    
        self.delay_embedding = EmbeddingModule(self.delay_nfeats, self.emb_size).to(device)
        self.edge_embedding = EdgeEmbeddingModule(self.edge_nfeats).to(device)
        self.service_embedding = EmbeddingModule(self.service_nfeats, self.emb_size).to(device)        

        self.conv_v_to_c = BipartiteGraphConvolution(self.emb_size, right_to_left=True, device=device).to(device)   
        self.conv_c_to_v = BipartiteGraphConvolution(self.emb_size, device=device).to(device)   
        

        self.theta_6 = nn.Linear(self.emb_size, self.emb_size, True).to(device)    
        nn.init.normal_(self.theta_6.weight, mean=0, std=0.01)        
        
        self.theta_7 = nn.Linear(self.emb_size, self.emb_size, True).to(device)    
        nn.init.normal_(self.theta_7.weight, mean=0, std=0.01)           
        
        self.theta_5 = nn.Linear(2 * self.emb_size, 1).to(device)
        nn.init.normal_(self.theta_5.weight, mean=0, std=0.01)  
        
        self.cat_67_ReLU = nn.ReLU().to(device)     
        

    def forward(self, obs): #obs
        #print('------forward-------------')
        #obs == [delay_feature, service_feature, edge_feature, edge_indice, adj]
        
        delay_feature = torch.Tensor(obs[0]).to(self.device)
        service_feature = torch.Tensor(obs[1]).to(self.device)
        edge_feature = torch.Tensor(obs[2]).to(self.device)
        edge_indice = torch.Tensor(obs[3]).type(torch.LongTensor).to(self.device)
        adj = torch.Tensor(obs[4]).to(self.device)
        delay_ones = torch.ones(len(delay_feature),1).to(self.device)
        service_ones = torch.ones(1, len(service_feature)).to(self.device)

        #print('========input ')
        #print('delay_feature ',delay_feature.shape)
        #print('service_feature ' ,service_feature.shape)
        #print('adj ',adj.shape)
        
        service_feature = self.service_embedding(service_feature)
        delay_feature = self.delay_embedding(delay_feature)
        edge_feature = self.edge_embedding(edge_feature)
        
    
        # Convolutions
        delay_feature = self.conv_v_to_c((
                delay_feature, edge_indice, edge_feature, service_feature, len(delay_feature)))
        service_feature = self.conv_c_to_v((
                delay_feature, edge_indice, edge_feature, service_feature, len(service_feature)))
        
        sum_service_feature = torch.matmul(service_ones, service_feature).expand(len(service_feature), self.emb_size)
        theta_6 = self.theta_6(sum_service_feature)
        theta_7 = self.theta_7(service_feature)
        c67 = torch.cat((theta_6, theta_7), dim=-1)
        c67 = self.cat_67_ReLU(c67)
        
        Q = self.theta_5(c67)

        return Q      
'''
class my_S2V_QN_scheme3(nn.Module):
    def __init__(self, device):

        super(my_S2V_QN_scheme3, self).__init__()
        self.device = device
        self.emb_size = 64
        self.delay_nfeats = 1
        self.edge_nfeats = 1
        self.service_nfeats = 6
        
        self.delay_embedding = EmbeddingModule(self.delay_nfeats, self.emb_size).to(device)
        #self.edge_embedding = EdgeEmbeddingModule(self.edge_nfeats).to(device)
        self.service_embedding = EmbeddingModule(self.service_nfeats, self.emb_size).to(device)        
        ########delay
        self.theta_1_delay = nn.Linear(self.emb_size, self.emb_size, True).to(device)
        nn.init.normal_(self.theta_1_delay.weight, mean=0, std=0.01)
    
        self.theta_2_delay = nn.Linear(self.emb_size, self.emb_size, True).to(device)
        nn.init.normal_(self.theta_2_delay.weight, mean=0, std=0.01)
    
        self.theta_3_delay = nn.Linear(self.emb_size, self.emb_size, True).to(device)    
        nn.init.normal_(self.theta_3_delay.weight, mean=0, std=0.01)      

        ########service
        self.theta_1_service = nn.Linear(self.emb_size, self.emb_size, True).to(device)
        nn.init.normal_(self.theta_1_service.weight, mean=0, std=0.01)
    
        self.theta_2_service = nn.Linear(self.emb_size, self.emb_size, True).to(device)
        nn.init.normal_(self.theta_2_service.weight, mean=0, std=0.01)
    
        self.theta_3_service = nn.Linear(self.emb_size, self.emb_size, True).to(device)    
        nn.init.normal_(self.theta_3_service.weight, mean=0, std=0.01)         
        ########
        self.theta_4 = nn.Sequential().to(device) 
        theta_4_1 = nn.Linear(1, self.emb_size, True).to(device)
        nn.init.normal_(theta_4_1.weight, mean=0, std=0.01)
        theta_4_2 = nn.ReLU().to(device)        
        self.theta_4.add_module('linear', theta_4_1)
        self.theta_4.add_module('relu', theta_4_2)
    
        self.sum_theta_ReLU = nn.ReLU().to(device)      
    
        self.theta_6 = nn.Linear(self.emb_size, self.emb_size, True).to(device)    
        nn.init.normal_(self.theta_6.weight, mean=0, std=0.01)        
    
        self.theta_7 = nn.Linear(self.emb_size, self.emb_size, True).to(device)    
        nn.init.normal_(self.theta_7.weight, mean=0, std=0.01)           
    
        self.theta_5 = nn.Linear(2 * self.emb_size, 1).to(device)
        nn.init.normal_(self.theta_5.weight, mean=0, std=0.01)  
    
        self.cat_67_ReLU = nn.ReLU().to(device)         
    
    def get_delay_feature(self, service_feature, delay_feature, adj):
        adj_t = adj.t()
        service_ones = torch.ones(len(service_feature), 1).to(self.device)        
        theta_1 = self.theta_1_delay(delay_feature)
    
        sum_neighbors_delay = torch.matmul(adj_t, service_feature)
        theta_2 = self.theta_2_delay(sum_neighbors_delay)
    
        weight_one = torch.Tensor([[1]]).to(self.device)
        theta_4 = self.theta_4(weight_one)
        sum_edge_weight = torch.matmul(adj_t, service_ones)
        sum_edge_weight = torch.matmul(sum_edge_weight, theta_4)  
    
        theta_3 = self.theta_3_delay(sum_edge_weight)     
    
        sum_theta = torch.add(theta_1, theta_2)
        sum_theta = torch.add(sum_theta, theta_3)
        sum_theta = self.sum_theta_ReLU(sum_theta)  
        return sum_theta
    
    def get_service_feature(self, service_feature, delay_feature, adj):
        delay_ones = torch.ones(len(delay_feature),1).to(self.device)
        theta_1 = self.theta_1_service(service_feature)
    
        sum_neighbors_delay = torch.matmul(adj, delay_feature)
        theta_2 = self.theta_2_service(sum_neighbors_delay)
    
        weight_one = torch.Tensor([[1]]).to(self.device)
        theta_4 = self.theta_4(weight_one)
        sum_edge_weight = torch.matmul(adj, delay_ones)
        sum_edge_weight = torch.matmul(sum_edge_weight, theta_4)  
    
        theta_3 = self.theta_3_service(sum_edge_weight)     
    
        sum_theta = torch.add(theta_1, theta_2)
        sum_theta = torch.add(sum_theta, theta_3)
        sum_theta = self.sum_theta_ReLU(sum_theta)
        return sum_theta
        
    
    def forward(self, obs): #obs
        #print('------forward-------------')
        #obs == [delay_feature, service_feature, edge_feature, edge_indice, adj]
        
        delay_feature = torch.Tensor(obs[0]).to(self.device)
        service_feature = torch.Tensor(obs[1]).to(self.device)
        adj = torch.Tensor(obs[4]).to(self.device)

        #print('========input ')
        #print('delay_feature ',delay_feature.shape)
        #print('service_feature ' ,service_feature.shape)
        #print('adj ',adj.shape)
        ###########embedding
        service_feature = self.service_embedding(service_feature)
        delay_feature = self.delay_embedding(delay_feature)
        ##########
        #for i in range(4):
        delay_feature = self.get_delay_feature(service_feature, delay_feature, adj)
        service_feature = self.get_service_feature(service_feature, delay_feature, adj)
        ##########
        service_ones = torch.ones(1, len(service_feature)).to(self.device)
        sum_service_feature = torch.matmul(service_ones, service_feature).expand(len(service_feature), self.emb_size)
        theta_6 = self.theta_6(sum_service_feature)
        theta_7 = self.theta_7(service_feature)
        c67 = torch.cat((theta_6, theta_7), dim=-1)
        c67 = self.cat_67_ReLU(c67)
        
        Q = self.theta_5(c67)
        return Q    





'''
obs=  [[[59.281471432286374], [75.42461461294737], [59.10883453486757], [60.420767772552935], [71.8079841718902], [63.26747142426144], [60.887928644371165], [60.853865560482944], [55.66484099122922], [60.78187019569785], [59.10883453486757], [60.78187019569785], [58.67573674693756], [71.3748863839602], [61.06056554178997], [64.4704960015401], [63.4349416472311], [79.57808482789208], [63.26230474981229], [64.57423798749765], [75.96145438683492], [67.42094163920618], [65.04139885931588], [65.00733577542766], [59.81831120617394], [64.93534041064258], [63.26230474981229], [64.93534041064258], [62.82920696188229], [75.52835659890494], [65.21403575673469], [68.62396621648483], [69.20026719842986], [85.34341037909084], [69.02763030101104], [70.33956353869641], [81.72677993803367], [73.18626719040493], [70.80672441051463], [70.77266132662642], [65.58363675737269], [70.70066596184134], [69.02763030101104], [70.70066596184134], [68.59453251308105], [81.29368215010369], [70.97936130793344], [74.38929176768357], [69.20026719842986], [85.34341037909084], [69.02763030101104], [70.33956353869641], [81.72677993803367], [73.18626719040493], [70.80672441051463], [70.77266132662642], [65.58363675737269], [70.70066596184134], [69.02763030101104], [70.70066596184134], [68.59453251308105], [81.29368215010369], [70.97936130793344], [74.38929176768357], [58.882628490266846], [75.02577167092785], [58.709991592848034], [60.02192483053342], [71.40914122987068], [62.868628482241924], [60.489085702351645], [60.45502261846342], [55.26599804920969], [60.383027253678335], [58.709991592848034], [60.383027253678335], [58.276893804918046], [70.97604344194067], [60.66172259977046], [64.07165305952057], [69.20026719842986], [85.34341037909084], [69.02763030101104], [70.33956353869641], [81.72677993803367], [73.18626719040493], [70.80672441051463], [70.77266132662642], [65.58363675737269], [70.70066596184134], [69.02763030101104], [70.70066596184134], [68.59453251308105], [81.29368215010369], [70.97936130793344], [74.38929176768357], [87.07321226951265], [103.21635545017364], [86.90057537209384], [88.21250860977922], [99.59972500911647], [91.05921226148772], [88.67966948159743], [88.64560639770922], [83.45658182845548], [88.57361103292412], [86.90057537209384], [88.57361103292412], [86.46747758416384], [99.16662722118647], [88.85230637901624], [92.26223683876637], [87.07321226951265], [103.21635545017364], [86.90057537209384], [88.21250860977922], [99.59972500911647], [91.05921226148772], [88.67966948159743], [88.64560639770922], [83.45658182845548], [88.57361103292412], [86.90057537209384], [88.57361103292412], [86.46747758416384], [99.16662722118647], [88.85230637901624], [92.26223683876637], [59.281471432286374], [75.42461461294737], [59.10883453486757], [60.420767772552935], [71.8079841718902], [63.26747142426144], [60.887928644371165], [60.853865560482944], [55.66484099122922], [60.78187019569785], [59.10883453486757], [60.78187019569785], [58.67573674693756], [71.3748863839602], [61.06056554178997], [64.4704960015401], [87.07321226951265], [103.21635545017364], [86.90057537209384], [88.21250860977922], [99.59972500911647], [91.05921226148772], [88.67966948159743], [88.64560639770922], [83.45658182845548], [88.57361103292412], [86.90057537209384], [88.57361103292412], [86.46747758416384], [99.16662722118647], [88.85230637901624], [92.26223683876637], [58.882628490266846], [75.02577167092785], [58.709991592848034], [60.02192483053342], [71.40914122987068], [62.868628482241924], [60.489085702351645], [60.45502261846342], [55.26599804920969], [60.383027253678335], [58.709991592848034], [60.383027253678335], [58.276893804918046], [70.97604344194067], [60.66172259977046], [64.07165305952057], [87.07321226951265], [103.21635545017364], [86.90057537209384], [88.21250860977922], [99.59972500911647], [91.05921226148772], [88.67966948159743], [88.64560639770922], [83.45658182845548], [88.57361103292412], [86.90057537209384], [88.57361103292412], [86.46747758416384], [99.16662722118647], [88.85230637901624], [92.26223683876637], [87.07321226951265], [103.21635545017364], [86.90057537209384], [88.21250860977922], [99.59972500911647], [91.05921226148772], [88.67966948159743], [88.64560639770922], [83.45658182845548], [88.57361103292412], [86.90057537209384], [88.57361103292412], [86.46747758416384], [99.16662722118647], [88.85230637901624], [92.26223683876637], [59.281471432286374], [75.42461461294737], [59.10883453486757], [60.420767772552935], [71.8079841718902], [63.26747142426144], [60.887928644371165], [60.853865560482944], [55.66484099122922], [60.78187019569785], [59.10883453486757], [60.78187019569785], [58.67573674693756], [71.3748863839602], [61.06056554178997], [64.4704960015401], [87.07321226951265], [103.21635545017364], [86.90057537209384], [88.21250860977922], [99.59972500911647], [91.05921226148772], [88.67966948159743], [88.64560639770922], [83.45658182845548], [88.57361103292412], [86.90057537209384], [88.57361103292412], [86.46747758416384], [99.16662722118647], [88.85230637901624], [92.26223683876637], [59.281471432286374], [75.42461461294737], [59.10883453486757], [60.420767772552935], [71.8079841718902], [63.26747142426144], [60.887928644371165], [60.853865560482944], [55.66484099122922], [60.78187019569785], [59.10883453486757], [60.78187019569785], [58.67573674693756], [71.3748863839602], [61.06056554178997], [64.4704960015401]], [[223.0, 111.0, 3.0, 4.078900086391093, 3.281910943134997, 2.0], [330.0, 119.0, 3.0, 31.87064092361737, 29.00567255030096, 5.0], [162.0, 105.0, 2.0, 13.997695852534568, 12.348136668376219, 5.0], [108.0, 103.0, 2.0, 3.6800571443715695, 3.17221303773792, 5.0], [53.0, 111.0, 1.0, 8.232370301335818, 7.687855910667581, 4.0], [876.0, 101.0, 9.0, 26.617886947174238, 22.319899220176886, 3.0], [876.0, 119.0, 8.0, 9.966399504039911, 7.542308808583009, 4.0], [876.0, 116.0, 8.0, 15.889299235020793, 12.775367747052021, 4.0], [876.0, 107.0, 9.0, 8.187305394536308, 5.85656458299977, 1.0], [876.0, 114.0, 8.0, 24.330448575197302, 20.597421073349413, 5.0], [876.0, 117.0, 8.0, 13.376329963790035, 10.519190688667264, 3.0], [175.20000000000002, 119.0, 2.0, 9.940341062032624, 8.691561826193817, 15.0], [175.20000000000002, 119.0, 2.0, 9.940341062032624, 8.691561826193817, 9.0], [175.20000000000002, 101.0, 2.0, 30.063556204467442, 27.502547809543273, 3.0], [175.20000000000002, 120.0, 2.0, 9.507243274102624, 8.301900810063948, 9.0], [175.20000000000002, 116.0, 2.0, 11.440739825444098, 10.048656800288018, 2.0], [700.8000000000001, 114.0, 7.0, 6.840407501094954, 4.982442041731108, 1.0], [350.40000000000003, 104.0, 4.0, 10.284401044733295, 8.295244151148198, 4.0], [525.6, 119.0, 5.0, 10.457037942152112, 8.350684900877939, 1.0], [350.40000000000003, 104.0, 4.0, 10.284401044733295, 8.295244151148198, 5.0], [175.20000000000002, 111.0, 2.0, 14.876135722057176, 13.192980900780924, 2.0]], [[1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0]], [[0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 4.0, 4.0, 5.0, 5.0, 5.0, 5.0, 5.0, 6.0, 6.0, 6.0, 6.0, 6.0, 7.0, 7.0, 7.0, 7.0, 7.0, 8.0, 8.0, 8.0, 8.0, 8.0, 9.0, 9.0, 9.0, 9.0, 9.0, 10.0, 10.0, 10.0, 10.0, 10.0, 11.0, 11.0, 11.0, 11.0, 11.0, 12.0, 12.0, 12.0, 12.0, 12.0, 13.0, 13.0, 13.0, 13.0, 13.0, 14.0, 14.0, 14.0, 14.0, 14.0, 15.0, 15.0, 15.0, 15.0, 15.0, 16.0, 16.0, 16.0, 16.0, 16.0, 17.0, 17.0, 17.0, 17.0, 17.0, 18.0, 18.0, 18.0, 18.0, 18.0, 19.0, 19.0, 19.0, 19.0, 19.0, 20.0, 20.0, 20.0, 20.0, 20.0, 21.0, 21.0, 21.0, 21.0, 21.0, 22.0, 22.0, 22.0, 22.0, 22.0, 23.0, 23.0, 23.0, 23.0, 23.0, 24.0, 24.0, 24.0, 24.0, 24.0, 25.0, 25.0, 25.0, 25.0, 25.0, 26.0, 26.0, 26.0, 26.0, 26.0, 27.0, 27.0, 27.0, 27.0, 27.0, 28.0, 28.0, 28.0, 28.0, 28.0, 29.0, 29.0, 29.0, 29.0, 29.0, 30.0, 30.0, 30.0, 30.0, 30.0, 31.0, 31.0, 31.0, 31.0, 31.0, 32.0, 32.0, 32.0, 32.0, 32.0, 33.0, 33.0, 33.0, 33.0, 33.0, 34.0, 34.0, 34.0, 34.0, 34.0, 35.0, 35.0, 35.0, 35.0, 35.0, 36.0, 36.0, 36.0, 36.0, 36.0, 37.0, 37.0, 37.0, 37.0, 37.0, 38.0, 38.0, 38.0, 38.0, 38.0, 39.0, 39.0, 39.0, 39.0, 39.0, 40.0, 40.0, 40.0, 40.0, 40.0, 41.0, 41.0, 41.0, 41.0, 41.0, 42.0, 42.0, 42.0, 42.0, 42.0, 43.0, 43.0, 43.0, 43.0, 43.0, 44.0, 44.0, 44.0, 44.0, 44.0, 45.0, 45.0, 45.0, 45.0, 45.0, 46.0, 46.0, 46.0, 46.0, 46.0, 47.0, 47.0, 47.0, 47.0, 47.0, 48.0, 48.0, 48.0, 48.0, 48.0, 49.0, 49.0, 49.0, 49.0, 49.0, 50.0, 50.0, 50.0, 50.0, 50.0, 51.0, 51.0, 51.0, 51.0, 51.0, 52.0, 52.0, 52.0, 52.0, 52.0, 53.0, 53.0, 53.0, 53.0, 53.0, 54.0, 54.0, 54.0, 54.0, 54.0, 55.0, 55.0, 55.0, 55.0, 55.0, 56.0, 56.0, 56.0, 56.0, 56.0, 57.0, 57.0, 57.0, 57.0, 57.0, 58.0, 58.0, 58.0, 58.0, 58.0, 59.0, 59.0, 59.0, 59.0, 59.0, 60.0, 60.0, 60.0, 60.0, 60.0, 61.0, 61.0, 61.0, 61.0, 61.0, 62.0, 62.0, 62.0, 62.0, 62.0, 63.0, 63.0, 63.0, 63.0, 63.0, 64.0, 64.0, 64.0, 64.0, 64.0, 65.0, 65.0, 65.0, 65.0, 65.0, 66.0, 66.0, 66.0, 66.0, 66.0, 67.0, 67.0, 67.0, 67.0, 67.0, 68.0, 68.0, 68.0, 68.0, 68.0, 69.0, 69.0, 69.0, 69.0, 69.0, 70.0, 70.0, 70.0, 70.0, 70.0, 71.0, 71.0, 71.0, 71.0, 71.0, 72.0, 72.0, 72.0, 72.0, 72.0, 73.0, 73.0, 73.0, 73.0, 73.0, 74.0, 74.0, 74.0, 74.0, 74.0, 75.0, 75.0, 75.0, 75.0, 75.0, 76.0, 76.0, 76.0, 76.0, 76.0, 77.0, 77.0, 77.0, 77.0, 77.0, 78.0, 78.0, 78.0, 78.0, 78.0, 79.0, 79.0, 79.0, 79.0, 79.0, 80.0, 80.0, 80.0, 80.0, 80.0, 81.0, 81.0, 81.0, 81.0, 81.0, 82.0, 82.0, 82.0, 82.0, 82.0, 83.0, 83.0, 83.0, 83.0, 83.0, 84.0, 84.0, 84.0, 84.0, 84.0, 85.0, 85.0, 85.0, 85.0, 85.0, 86.0, 86.0, 86.0, 86.0, 86.0, 87.0, 87.0, 87.0, 87.0, 87.0, 88.0, 88.0, 88.0, 88.0, 88.0, 89.0, 89.0, 89.0, 89.0, 89.0, 90.0, 90.0, 90.0, 90.0, 90.0, 91.0, 91.0, 91.0, 91.0, 91.0, 92.0, 92.0, 92.0, 92.0, 92.0, 93.0, 93.0, 93.0, 93.0, 93.0, 94.0, 94.0, 94.0, 94.0, 94.0, 95.0, 95.0, 95.0, 95.0, 95.0, 96.0, 96.0, 96.0, 96.0, 96.0, 97.0, 97.0, 97.0, 97.0, 97.0, 98.0, 98.0, 98.0, 98.0, 98.0, 99.0, 99.0, 99.0, 99.0, 99.0, 100.0, 100.0, 100.0, 100.0, 100.0, 101.0, 101.0, 101.0, 101.0, 101.0, 102.0, 102.0, 102.0, 102.0, 102.0, 103.0, 103.0, 103.0, 103.0, 103.0, 104.0, 104.0, 104.0, 104.0, 104.0, 105.0, 105.0, 105.0, 105.0, 105.0, 106.0, 106.0, 106.0, 106.0, 106.0, 107.0, 107.0, 107.0, 107.0, 107.0, 108.0, 108.0, 108.0, 108.0, 108.0, 109.0, 109.0, 109.0, 109.0, 109.0, 110.0, 110.0, 110.0, 110.0, 110.0, 111.0, 111.0, 111.0, 111.0, 111.0, 112.0, 112.0, 112.0, 112.0, 112.0, 113.0, 113.0, 113.0, 113.0, 113.0, 114.0, 114.0, 114.0, 114.0, 114.0, 115.0, 115.0, 115.0, 115.0, 115.0, 116.0, 116.0, 116.0, 116.0, 116.0, 117.0, 117.0, 117.0, 117.0, 117.0, 118.0, 118.0, 118.0, 118.0, 118.0, 119.0, 119.0, 119.0, 119.0, 119.0, 120.0, 120.0, 120.0, 120.0, 120.0, 121.0, 121.0, 121.0, 121.0, 121.0, 122.0, 122.0, 122.0, 122.0, 122.0, 123.0, 123.0, 123.0, 123.0, 123.0, 124.0, 124.0, 124.0, 124.0, 124.0, 125.0, 125.0, 125.0, 125.0, 125.0, 126.0, 126.0, 126.0, 126.0, 126.0, 127.0, 127.0, 127.0, 127.0, 127.0, 128.0, 128.0, 128.0, 128.0, 128.0, 129.0, 129.0, 129.0, 129.0, 129.0, 130.0, 130.0, 130.0, 130.0, 130.0, 131.0, 131.0, 131.0, 131.0, 131.0, 132.0, 132.0, 132.0, 132.0, 132.0, 133.0, 133.0, 133.0, 133.0, 133.0, 134.0, 134.0, 134.0, 134.0, 134.0, 135.0, 135.0, 135.0, 135.0, 135.0, 136.0, 136.0, 136.0, 136.0, 136.0, 137.0, 137.0, 137.0, 137.0, 137.0, 138.0, 138.0, 138.0, 138.0, 138.0, 139.0, 139.0, 139.0, 139.0, 139.0, 140.0, 140.0, 140.0, 140.0, 140.0, 141.0, 141.0, 141.0, 141.0, 141.0, 142.0, 142.0, 142.0, 142.0, 142.0, 143.0, 143.0, 143.0, 143.0, 143.0, 144.0, 144.0, 144.0, 144.0, 144.0, 145.0, 145.0, 145.0, 145.0, 145.0, 146.0, 146.0, 146.0, 146.0, 146.0, 147.0, 147.0, 147.0, 147.0, 147.0, 148.0, 148.0, 148.0, 148.0, 148.0, 149.0, 149.0, 149.0, 149.0, 149.0, 150.0, 150.0, 150.0, 150.0, 150.0, 151.0, 151.0, 151.0, 151.0, 151.0, 152.0, 152.0, 152.0, 152.0, 152.0, 153.0, 153.0, 153.0, 153.0, 153.0, 154.0, 154.0, 154.0, 154.0, 154.0, 155.0, 155.0, 155.0, 155.0, 155.0, 156.0, 156.0, 156.0, 156.0, 156.0, 157.0, 157.0, 157.0, 157.0, 157.0, 158.0, 158.0, 158.0, 158.0, 158.0, 159.0, 159.0, 159.0, 159.0, 159.0, 160.0, 160.0, 160.0, 160.0, 160.0, 161.0, 161.0, 161.0, 161.0, 161.0, 162.0, 162.0, 162.0, 162.0, 162.0, 163.0, 163.0, 163.0, 163.0, 163.0, 164.0, 164.0, 164.0, 164.0, 164.0, 165.0, 165.0, 165.0, 165.0, 165.0, 166.0, 166.0, 166.0, 166.0, 166.0, 167.0, 167.0, 167.0, 167.0, 167.0, 168.0, 168.0, 168.0, 168.0, 168.0, 169.0, 169.0, 169.0, 169.0, 169.0, 170.0, 170.0, 170.0, 170.0, 170.0, 171.0, 171.0, 171.0, 171.0, 171.0, 172.0, 172.0, 172.0, 172.0, 172.0, 173.0, 173.0, 173.0, 173.0, 173.0, 174.0, 174.0, 174.0, 174.0, 174.0, 175.0, 175.0, 175.0, 175.0, 175.0, 176.0, 176.0, 176.0, 176.0, 176.0, 177.0, 177.0, 177.0, 177.0, 177.0, 178.0, 178.0, 178.0, 178.0, 178.0, 179.0, 179.0, 179.0, 179.0, 179.0, 180.0, 180.0, 180.0, 180.0, 180.0, 181.0, 181.0, 181.0, 181.0, 181.0, 182.0, 182.0, 182.0, 182.0, 182.0, 183.0, 183.0, 183.0, 183.0, 183.0, 184.0, 184.0, 184.0, 184.0, 184.0, 185.0, 185.0, 185.0, 185.0, 185.0, 186.0, 186.0, 186.0, 186.0, 186.0, 187.0, 187.0, 187.0, 187.0, 187.0, 188.0, 188.0, 188.0, 188.0, 188.0, 189.0, 189.0, 189.0, 189.0, 189.0, 190.0, 190.0, 190.0, 190.0, 190.0, 191.0, 191.0, 191.0, 191.0, 191.0, 192.0, 192.0, 192.0, 192.0, 192.0, 193.0, 193.0, 193.0, 193.0, 193.0, 194.0, 194.0, 194.0, 194.0, 194.0, 195.0, 195.0, 195.0, 195.0, 195.0, 196.0, 196.0, 196.0, 196.0, 196.0, 197.0, 197.0, 197.0, 197.0, 197.0, 198.0, 198.0, 198.0, 198.0, 198.0, 199.0, 199.0, 199.0, 199.0, 199.0, 200.0, 200.0, 200.0, 200.0, 200.0, 201.0, 201.0, 201.0, 201.0, 201.0, 202.0, 202.0, 202.0, 202.0, 202.0, 203.0, 203.0, 203.0, 203.0, 203.0, 204.0, 204.0, 204.0, 204.0, 204.0, 205.0, 205.0, 205.0, 205.0, 205.0, 206.0, 206.0, 206.0, 206.0, 206.0, 207.0, 207.0, 207.0, 207.0, 207.0, 208.0, 208.0, 208.0, 208.0, 208.0, 209.0, 209.0, 209.0, 209.0, 209.0, 210.0, 210.0, 210.0, 210.0, 210.0, 211.0, 211.0, 211.0, 211.0, 211.0, 212.0, 212.0, 212.0, 212.0, 212.0, 213.0, 213.0, 213.0, 213.0, 213.0, 214.0, 214.0, 214.0, 214.0, 214.0, 215.0, 215.0, 215.0, 215.0, 215.0, 216.0, 216.0, 216.0, 216.0, 216.0, 217.0, 217.0, 217.0, 217.0, 217.0, 218.0, 218.0, 218.0, 218.0, 218.0, 219.0, 219.0, 219.0, 219.0, 219.0, 220.0, 220.0, 220.0, 220.0, 220.0, 221.0, 221.0, 221.0, 221.0, 221.0, 222.0, 222.0, 222.0, 222.0, 222.0, 223.0, 223.0, 223.0, 223.0, 223.0, 224.0, 224.0, 224.0, 224.0, 224.0, 225.0, 225.0, 225.0, 225.0, 225.0, 226.0, 226.0, 226.0, 226.0, 226.0, 227.0, 227.0, 227.0, 227.0, 227.0, 228.0, 228.0, 228.0, 228.0, 228.0, 229.0, 229.0, 229.0, 229.0, 229.0, 230.0, 230.0, 230.0, 230.0, 230.0, 231.0, 231.0, 231.0, 231.0, 231.0, 232.0, 232.0, 232.0, 232.0, 232.0, 233.0, 233.0, 233.0, 233.0, 233.0, 234.0, 234.0, 234.0, 234.0, 234.0, 235.0, 235.0, 235.0, 235.0, 235.0, 236.0, 236.0, 236.0, 236.0, 236.0, 237.0, 237.0, 237.0, 237.0, 237.0, 238.0, 238.0, 238.0, 238.0, 238.0, 239.0, 239.0, 239.0, 239.0, 239.0, 240.0, 240.0, 240.0, 240.0, 240.0, 241.0, 241.0, 241.0, 241.0, 241.0, 242.0, 242.0, 242.0, 242.0, 242.0, 243.0, 243.0, 243.0, 243.0, 243.0, 244.0, 244.0, 244.0, 244.0, 244.0, 245.0, 245.0, 245.0, 245.0, 245.0, 246.0, 246.0, 246.0, 246.0, 246.0, 247.0, 247.0, 247.0, 247.0, 247.0, 248.0, 248.0, 248.0, 248.0, 248.0, 249.0, 249.0, 249.0, 249.0, 249.0, 250.0, 250.0, 250.0, 250.0, 250.0, 251.0, 251.0, 251.0, 251.0, 251.0, 252.0, 252.0, 252.0, 252.0, 252.0, 253.0, 253.0, 253.0, 253.0, 253.0, 254.0, 254.0, 254.0, 254.0, 254.0, 255.0, 255.0, 255.0, 255.0, 255.0], [0.0, 5.0, 8.0, 11.0, 18.0, 0.0, 5.0, 9.0, 11.0, 18.0, 0.0, 5.0, 8.0, 12.0, 17.0, 0.0, 5.0, 10.0, 14.0, 16.0, 0.0, 5.0, 9.0, 11.0, 16.0, 0.0, 5.0, 8.0, 14.0, 20.0, 0.0, 5.0, 6.0, 12.0, 19.0, 0.0, 5.0, 10.0, 11.0, 16.0, 0.0, 5.0, 8.0, 11.0, 16.0, 0.0, 5.0, 8.0, 15.0, 18.0, 0.0, 5.0, 8.0, 11.0, 19.0, 0.0, 5.0, 8.0, 15.0, 18.0, 0.0, 5.0, 8.0, 14.0, 17.0, 0.0, 5.0, 9.0, 14.0, 16.0, 0.0, 5.0, 6.0, 12.0, 18.0, 0.0, 5.0, 10.0, 12.0, 18.0, 4.0, 5.0, 8.0, 11.0, 18.0, 4.0, 5.0, 9.0, 11.0, 18.0, 4.0, 5.0, 8.0, 12.0, 17.0, 4.0, 5.0, 10.0, 14.0, 16.0, 4.0, 5.0, 9.0, 11.0, 16.0, 4.0, 5.0, 8.0, 14.0, 20.0, 4.0, 5.0, 6.0, 12.0, 19.0, 4.0, 5.0, 10.0, 11.0, 16.0, 4.0, 5.0, 8.0, 11.0, 16.0, 4.0, 5.0, 8.0, 15.0, 18.0, 4.0, 5.0, 8.0, 11.0, 19.0, 4.0, 5.0, 8.0, 15.0, 18.0, 4.0, 5.0, 8.0, 14.0, 17.0, 4.0, 5.0, 9.0, 14.0, 16.0, 4.0, 5.0, 6.0, 12.0, 18.0, 4.0, 5.0, 10.0, 12.0, 18.0, 2.0, 5.0, 8.0, 11.0, 18.0, 2.0, 5.0, 9.0, 11.0, 18.0, 2.0, 5.0, 8.0, 12.0, 17.0, 2.0, 5.0, 10.0, 14.0, 16.0, 2.0, 5.0, 9.0, 11.0, 16.0, 2.0, 5.0, 8.0, 14.0, 20.0, 2.0, 5.0, 6.0, 12.0, 19.0, 2.0, 5.0, 10.0, 11.0, 16.0, 2.0, 5.0, 8.0, 11.0, 16.0, 2.0, 5.0, 8.0, 15.0, 18.0, 2.0, 5.0, 8.0, 11.0, 19.0, 2.0, 5.0, 8.0, 15.0, 18.0, 2.0, 5.0, 8.0, 14.0, 17.0, 2.0, 5.0, 9.0, 14.0, 16.0, 2.0, 5.0, 6.0, 12.0, 18.0, 2.0, 5.0, 10.0, 12.0, 18.0, 2.0, 5.0, 8.0, 11.0, 18.0, 2.0, 5.0, 9.0, 11.0, 18.0, 2.0, 5.0, 8.0, 12.0, 17.0, 2.0, 5.0, 10.0, 14.0, 16.0, 2.0, 5.0, 9.0, 11.0, 16.0, 2.0, 5.0, 8.0, 14.0, 20.0, 2.0, 5.0, 6.0, 12.0, 19.0, 2.0, 5.0, 10.0, 11.0, 16.0, 2.0, 5.0, 8.0, 11.0, 16.0, 2.0, 5.0, 8.0, 15.0, 18.0, 2.0, 5.0, 8.0, 11.0, 19.0, 2.0, 5.0, 8.0, 15.0, 18.0, 2.0, 5.0, 8.0, 14.0, 17.0, 2.0, 5.0, 9.0, 14.0, 16.0, 2.0, 5.0, 6.0, 12.0, 18.0, 2.0, 5.0, 10.0, 12.0, 18.0, 3.0, 5.0, 8.0, 11.0, 18.0, 3.0, 5.0, 9.0, 11.0, 18.0, 3.0, 5.0, 8.0, 12.0, 17.0, 3.0, 5.0, 10.0, 14.0, 16.0, 3.0, 5.0, 9.0, 11.0, 16.0, 3.0, 5.0, 8.0, 14.0, 20.0, 3.0, 5.0, 6.0, 12.0, 19.0, 3.0, 5.0, 10.0, 11.0, 16.0, 3.0, 5.0, 8.0, 11.0, 16.0, 3.0, 5.0, 8.0, 15.0, 18.0, 3.0, 5.0, 8.0, 11.0, 19.0, 3.0, 5.0, 8.0, 15.0, 18.0, 3.0, 5.0, 8.0, 14.0, 17.0, 3.0, 5.0, 9.0, 14.0, 16.0, 3.0, 5.0, 6.0, 12.0, 18.0, 3.0, 5.0, 10.0, 12.0, 18.0, 2.0, 5.0, 8.0, 11.0, 18.0, 2.0, 5.0, 9.0, 11.0, 18.0, 2.0, 5.0, 8.0, 12.0, 17.0, 2.0, 5.0, 10.0, 14.0, 16.0, 2.0, 5.0, 9.0, 11.0, 16.0, 2.0, 5.0, 8.0, 14.0, 20.0, 2.0, 5.0, 6.0, 12.0, 19.0, 2.0, 5.0, 10.0, 11.0, 16.0, 2.0, 5.0, 8.0, 11.0, 16.0, 2.0, 5.0, 8.0, 15.0, 18.0, 2.0, 5.0, 8.0, 11.0, 19.0, 2.0, 5.0, 8.0, 15.0, 18.0, 2.0, 5.0, 8.0, 14.0, 17.0, 2.0, 5.0, 9.0, 14.0, 16.0, 2.0, 5.0, 6.0, 12.0, 18.0, 2.0, 5.0, 10.0, 12.0, 18.0, 1.0, 5.0, 8.0, 11.0, 18.0, 1.0, 5.0, 9.0, 11.0, 18.0, 1.0, 5.0, 8.0, 12.0, 17.0, 1.0, 5.0, 10.0, 14.0, 16.0, 1.0, 5.0, 9.0, 11.0, 16.0, 1.0, 5.0, 8.0, 14.0, 20.0, 1.0, 5.0, 6.0, 12.0, 19.0, 1.0, 5.0, 10.0, 11.0, 16.0, 1.0, 5.0, 8.0, 11.0, 16.0, 1.0, 5.0, 8.0, 15.0, 18.0, 1.0, 5.0, 8.0, 11.0, 19.0, 1.0, 5.0, 8.0, 15.0, 18.0, 1.0, 5.0, 8.0, 14.0, 17.0, 1.0, 5.0, 9.0, 14.0, 16.0, 1.0, 5.0, 6.0, 12.0, 18.0, 1.0, 5.0, 10.0, 12.0, 18.0, 1.0, 5.0, 8.0, 11.0, 18.0, 1.0, 5.0, 9.0, 11.0, 18.0, 1.0, 5.0, 8.0, 12.0, 17.0, 1.0, 5.0, 10.0, 14.0, 16.0, 1.0, 5.0, 9.0, 11.0, 16.0, 1.0, 5.0, 8.0, 14.0, 20.0, 1.0, 5.0, 6.0, 12.0, 19.0, 1.0, 5.0, 10.0, 11.0, 16.0, 1.0, 5.0, 8.0, 11.0, 16.0, 1.0, 5.0, 8.0, 15.0, 18.0, 1.0, 5.0, 8.0, 11.0, 19.0, 1.0, 5.0, 8.0, 15.0, 18.0, 1.0, 5.0, 8.0, 14.0, 17.0, 1.0, 5.0, 9.0, 14.0, 16.0, 1.0, 5.0, 6.0, 12.0, 18.0, 1.0, 5.0, 10.0, 12.0, 18.0, 0.0, 5.0, 8.0, 11.0, 18.0, 0.0, 5.0, 9.0, 11.0, 18.0, 0.0, 5.0, 8.0, 12.0, 17.0, 0.0, 5.0, 10.0, 14.0, 16.0, 0.0, 5.0, 9.0, 11.0, 16.0, 0.0, 5.0, 8.0, 14.0, 20.0, 0.0, 5.0, 6.0, 12.0, 19.0, 0.0, 5.0, 10.0, 11.0, 16.0, 0.0, 5.0, 8.0, 11.0, 16.0, 0.0, 5.0, 8.0, 15.0, 18.0, 0.0, 5.0, 8.0, 11.0, 19.0, 0.0, 5.0, 8.0, 15.0, 18.0, 0.0, 5.0, 8.0, 14.0, 17.0, 0.0, 5.0, 9.0, 14.0, 16.0, 0.0, 5.0, 6.0, 12.0, 18.0, 0.0, 5.0, 10.0, 12.0, 18.0, 1.0, 5.0, 8.0, 11.0, 18.0, 1.0, 5.0, 9.0, 11.0, 18.0, 1.0, 5.0, 8.0, 12.0, 17.0, 1.0, 5.0, 10.0, 14.0, 16.0, 1.0, 5.0, 9.0, 11.0, 16.0, 1.0, 5.0, 8.0, 14.0, 20.0, 1.0, 5.0, 6.0, 12.0, 19.0, 1.0, 5.0, 10.0, 11.0, 16.0, 1.0, 5.0, 8.0, 11.0, 16.0, 1.0, 5.0, 8.0, 15.0, 18.0, 1.0, 5.0, 8.0, 11.0, 19.0, 1.0, 5.0, 8.0, 15.0, 18.0, 1.0, 5.0, 8.0, 14.0, 17.0, 1.0, 5.0, 9.0, 14.0, 16.0, 1.0, 5.0, 6.0, 12.0, 18.0, 1.0, 5.0, 10.0, 12.0, 18.0, 3.0, 5.0, 8.0, 11.0, 18.0, 3.0, 5.0, 9.0, 11.0, 18.0, 3.0, 5.0, 8.0, 12.0, 17.0, 3.0, 5.0, 10.0, 14.0, 16.0, 3.0, 5.0, 9.0, 11.0, 16.0, 3.0, 5.0, 8.0, 14.0, 20.0, 3.0, 5.0, 6.0, 12.0, 19.0, 3.0, 5.0, 10.0, 11.0, 16.0, 3.0, 5.0, 8.0, 11.0, 16.0, 3.0, 5.0, 8.0, 15.0, 18.0, 3.0, 5.0, 8.0, 11.0, 19.0, 3.0, 5.0, 8.0, 15.0, 18.0, 3.0, 5.0, 8.0, 14.0, 17.0, 3.0, 5.0, 9.0, 14.0, 16.0, 3.0, 5.0, 6.0, 12.0, 18.0, 3.0, 5.0, 10.0, 12.0, 18.0, 1.0, 5.0, 8.0, 11.0, 18.0, 1.0, 5.0, 9.0, 11.0, 18.0, 1.0, 5.0, 8.0, 12.0, 17.0, 1.0, 5.0, 10.0, 14.0, 16.0, 1.0, 5.0, 9.0, 11.0, 16.0, 1.0, 5.0, 8.0, 14.0, 20.0, 1.0, 5.0, 6.0, 12.0, 19.0, 1.0, 5.0, 10.0, 11.0, 16.0, 1.0, 5.0, 8.0, 11.0, 16.0, 1.0, 5.0, 8.0, 15.0, 18.0, 1.0, 5.0, 8.0, 11.0, 19.0, 1.0, 5.0, 8.0, 15.0, 18.0, 1.0, 5.0, 8.0, 14.0, 17.0, 1.0, 5.0, 9.0, 14.0, 16.0, 1.0, 5.0, 6.0, 12.0, 18.0, 1.0, 5.0, 10.0, 12.0, 18.0, 1.0, 5.0, 8.0, 11.0, 18.0, 1.0, 5.0, 9.0, 11.0, 18.0, 1.0, 5.0, 8.0, 12.0, 17.0, 1.0, 5.0, 10.0, 14.0, 16.0, 1.0, 5.0, 9.0, 11.0, 16.0, 1.0, 5.0, 8.0, 14.0, 20.0, 1.0, 5.0, 6.0, 12.0, 19.0, 1.0, 5.0, 10.0, 11.0, 16.0, 1.0, 5.0, 8.0, 11.0, 16.0, 1.0, 5.0, 8.0, 15.0, 18.0, 1.0, 5.0, 8.0, 11.0, 19.0, 1.0, 5.0, 8.0, 15.0, 18.0, 1.0, 5.0, 8.0, 14.0, 17.0, 1.0, 5.0, 9.0, 14.0, 16.0, 1.0, 5.0, 6.0, 12.0, 18.0, 1.0, 5.0, 10.0, 12.0, 18.0, 0.0, 5.0, 8.0, 11.0, 18.0, 0.0, 5.0, 9.0, 11.0, 18.0, 0.0, 5.0, 8.0, 12.0, 17.0, 0.0, 5.0, 10.0, 14.0, 16.0, 0.0, 5.0, 9.0, 11.0, 16.0, 0.0, 5.0, 8.0, 14.0, 20.0, 0.0, 5.0, 6.0, 12.0, 19.0, 0.0, 5.0, 10.0, 11.0, 16.0, 0.0, 5.0, 8.0, 11.0, 16.0, 0.0, 5.0, 8.0, 15.0, 18.0, 0.0, 5.0, 8.0, 11.0, 19.0, 0.0, 5.0, 8.0, 15.0, 18.0, 0.0, 5.0, 8.0, 14.0, 17.0, 0.0, 5.0, 9.0, 14.0, 16.0, 0.0, 5.0, 6.0, 12.0, 18.0, 0.0, 5.0, 10.0, 12.0, 18.0, 1.0, 5.0, 8.0, 11.0, 18.0, 1.0, 5.0, 9.0, 11.0, 18.0, 1.0, 5.0, 8.0, 12.0, 17.0, 1.0, 5.0, 10.0, 14.0, 16.0, 1.0, 5.0, 9.0, 11.0, 16.0, 1.0, 5.0, 8.0, 14.0, 20.0, 1.0, 5.0, 6.0, 12.0, 19.0, 1.0, 5.0, 10.0, 11.0, 16.0, 1.0, 5.0, 8.0, 11.0, 16.0, 1.0, 5.0, 8.0, 15.0, 18.0, 1.0, 5.0, 8.0, 11.0, 19.0, 1.0, 5.0, 8.0, 15.0, 18.0, 1.0, 5.0, 8.0, 14.0, 17.0, 1.0, 5.0, 9.0, 14.0, 16.0, 1.0, 5.0, 6.0, 12.0, 18.0, 1.0, 5.0, 10.0, 12.0, 18.0, 0.0, 5.0, 8.0, 11.0, 18.0, 0.0, 5.0, 9.0, 11.0, 18.0, 0.0, 5.0, 8.0, 12.0, 17.0, 0.0, 5.0, 10.0, 14.0, 16.0, 0.0, 5.0, 9.0, 11.0, 16.0, 0.0, 5.0, 8.0, 14.0, 20.0, 0.0, 5.0, 6.0, 12.0, 19.0, 0.0, 5.0, 10.0, 11.0, 16.0, 0.0, 5.0, 8.0, 11.0, 16.0, 0.0, 5.0, 8.0, 15.0, 18.0, 0.0, 5.0, 8.0, 11.0, 19.0, 0.0, 5.0, 8.0, 15.0, 18.0, 0.0, 5.0, 8.0, 14.0, 17.0, 0.0, 5.0, 9.0, 14.0, 16.0, 0.0, 5.0, 6.0, 12.0, 18.0, 0.0, 5.0, 10.0, 12.0, 18.0]], [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], [1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]]

device = torch.device('cpu')#('cuda' if torch.cuda.is_available() else 'cpu')
model = my_S2V_QN_scheme3_supervise(device).to(device)
para = torch.load('./supervise_checkpoint/model-819251/model-e=10000.pth')
model.load_state_dict(para)   
y = model(obs)
#print('output: ',y)

top5 = torch.topk(y, 5, dim=0)
print('top5 ',top5[0]) #值
print('top5 ',top5[1]) #下标
S = torch.nn.Softmax(dim=0)
softmaxy = S(top5[0])
print('softmaxy ',softmaxy)
y_zero = torch.zeros_like(y)
#print('y_zero ',y_zero)
index = top5[1].flatten()
k = 0
for i in index:
    y_zero[i][0] = softmaxy[k][0]
    k+=1
'''

#aaa

