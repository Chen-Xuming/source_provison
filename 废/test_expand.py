import torch
import torch.nn as nn
'''
c_ones = torch.Tensor([[1],[1],[1],[1],[1]])
print('c_ones ', c_ones.shape)

adj = torch.Tensor([[0,1,0,1,0],
           [1,1,0,1,1],
          [0,0,1,1,1],
            [1,1,0,0,0]])

theta_4 = nn.Linear(1, 64, True)
adj_list = []
for row in range(len(adj)):
    for col in range(len(adj[0])):
        #a = torch.unsqueeze(adj[row][col], 0)
        #print(a)
        adj_list.append( theta_4( torch.unsqueeze(adj[row][col], 0) ))

print('adj_list ',adj_list)
adj_list = torch.stack(adj_list) 
adj_list = torch.unsqueeze(adj_list, 0)
print('adj ',adj_list.shape)
adj_list= adj_list.reshape(len(adj), len(adj[0]), 64)
print('adj ',adj_list, adj_list.shape)
m = torch.matmul(adj_list, c_ones)
print('m ',m.shape)
'''


'''
tensor_one = torch.Tensor([[1]])
print('tensor_one ', tensor_one.shape)

adj = torch.Tensor([[0,1,0,1,0],
           [1,1,0,1,1],
          [0,0,1,1,1],
            [1,1,0,0,0]])

theta_4 = nn.Linear(1, 64, True)
t4 = theta_4(tensor_one)
print('t4 ',t4,t4.shape)

c_ones = torch.Tensor([[1],[1],[1],[1],[1]])
print('c_ones ', c_ones.shape)
m = torch.matmul(adj, c_ones)
print('m ',m, m.shape)

add = torch.matmul(m, t4)
print('add ',add, add.shape)
'''
'''
adj_list = []
for row in range(len(adj)):
    for col in range(len(adj[0])):
        #a = torch.unsqueeze(adj[row][col], 0)
        #print(a)
        adj_list.append( theta_4( torch.unsqueeze(adj[row][col], 0) ))

print('adj_list ',adj_list)
adj_list = torch.stack(adj_list) 
adj_list = torch.unsqueeze(adj_list, 0)
print('adj ',adj_list.shape)
adj_list= adj_list.reshape(len(adj), len(adj[0]), 64)
print('adj ',adj_list, adj_list.shape)
m = torch.matmul(adj_list, c_ones)
print('m ',m.shape)
'''


a = [1,2,3,4,5,6,7,7,8,43,3,6,7]
a = a[-5:]
print(a)