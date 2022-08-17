import numpy as np
import math
from decimal import Decimal 

class Instance:
    def __init__(self, id):
        self._id = id
        self._groups = []
    
    def __str__(self):
        return 'instance: {} groups: {}'.format(self._id, self._groups)

class ShareViewGroup:
    def __init__(self, id, instance_id):
        self._id = id
        self._instance_id = instance_id
        self._users = []

    def __str__(self):
        return 'instance: {} group: {} users: {}'.format(self._instance_id, self._id, self._users)

class User:
    def __init__(self, id, instance_id, group_id):
        self._id = id
        self._instance_id = instance_id
        self._group_id = group_id
        self._arrival_rate = 0
        self._service_a = 0
        self._service_b = 0
        self._service_c = 0
        self._service_r = 0
    
    def __str__(self):
        return 'instance: {} group: {} user: {} arrival_rate:{} service a:{} b:{} c:{} r:{}'.format(self._instance_id, 
                self._group_id, self._id, self._arrival_rate, 
                self._service_a, self._service_b, self._service_c, self._service_r)

class Service:
    def __init__(self, _id, _type):
        self._id = _id
        self._type = _type
        self._service_rate = 0
        self._arrival_rate = 0
        self._users = []
        self._group = set()
        self._price = 0
        self._num_server = 1
        self._queuing_delay = 0.
    
    def __str__(self):
        return 'service: {} {} service_rate: {} arrival_rate: {} users:{} group:{} price:{} num_server:{}'.format(self._type, self._id, self._service_rate,
        self._arrival_rate, self._users, self._group, self._price, self._num_server)
    
    def update_num_server(self, n):
        self._num_server = n
        self._queuing_delay = self.compute_queuing_delay(self._num_server)
        
    
    def initialize_num_server(self):
        num_server = self._num_server
        while num_server * self._service_rate <= self._arrival_rate:
            num_server += 1
        self.update_num_server(num_server)
    
    def reduction_of_delay_when_add_a_server(self):
        num_server = self._num_server + 1
        reduction = self._queuing_delay - self.compute_queuing_delay(num_server)
        return reduction

    def reduction_of_delay_when_add_some_server(self, n):
        if n == 0:
            return 0
        num_server = self._num_server + n
        reduction = self._queuing_delay - self.compute_queuing_delay(num_server)
        return reduction

    def compute_queuing_delay(self, num_server):
        '''
        lam = float(self._arrival_rate)
        mu = float(self._service_rate)
        c = num_server
        r = lam/mu
        rho = r/c
        #print('type {}:lam {} mu {} c {} r {} rho {}'.format(self._type,lam,mu,c,r,rho))
        assert rho < 1
        p0 = 1/(math.pow(r,c) / (float(math.factorial(c))*(1-rho)) + sum([math.pow(r,n)/float(math.factorial(n)) for n in range(0,c)]))
        queuing_delay = (math.pow(r,c) / (float(math.factorial(c)) * float(c) * mu * math.pow(1-rho,2))) * p0
        assert queuing_delay >= 0.
        queuing_delay_iteratively = self.compute_queuing_delay_iteratively(num_server)
        #print(queuing_delay_iteratively, '       ' ,queuing_delay)
        assert abs(queuing_delay_iteratively - queuing_delay) < 0.+ 1e-5
        '''
        queuing_delay_iteratively = self.compute_queuing_delay_iteratively(num_server)
        assert queuing_delay_iteratively >= 0.
        return queuing_delay_iteratively 
    
    
    def compute_queuing_delay_iteratively(self, num_server):
        lam = float(self._arrival_rate)
        mu = float(self._service_rate)
        c = num_server
        r = lam/mu
        rho = r/c
        assert rho < 1    
        
        p0c_2 = 1.
        n = 1
        p0c_1 = r/n
        n += 1
        while n <= c:
            p0c_2 += p0c_1            
            p0c_1 *= r/n
            n += 1
    
        p0 = 1/(p0c_1/(1-rho) + p0c_2)
        wq = p0c_1*p0/c/(mu * (1-rho)**2)
        #print('lam ',lam)
        #print('mu ',mu)
        #print('c ',c)
        #print('wq ',wq)
        assert wq >= 0.
        return wq    



class Env:
    def __init__(self, rng, num_instance, num_service_a, num_service_b, num_service_c, num_service_r, budget_addition, seed_sequence):
        self._seed_sequence = seed_sequence
        self._rng = rng
        self._num_instance = num_instance
        self._num_service_a = num_service_a
        self._num_service_b = num_service_b
        self._num_service_c = num_service_c
        self._num_service_r = num_service_r
        self._num_service = self._num_service_a + 1 + self._num_service_b + self._num_service_c + self._num_service_r

        self._num_group_per_instance = 25    # 每个instance的组数
        self._num_user_per_group = 4      # 每个组的用户数
        
        self._max_arrival_rate = 61     #61
        self._min_arrival_rate = 50
        
        self._max_service_rate = 601    #120
        self._min_service_rate = 500    #100
        
        self._max_price = 6
        self._min_price = 1
        
        self._trigger_probability = 0.2
        
        self._max_tx = 6
        self._min_tx = 1
        
        self._cost_budget = 0
        self._budget_addition = budget_addition
        
        self.initialize_user()
        self.correct_initialize_user()
        self.initialize_service()
        
        self.get_delay_without_duplicate()
        
        self.initialize_num_server()
        
        self.save_price_list()
        #self._print()
        
    
    def get_delay_without_duplicate(self):
        self._delay_without_duplicate = {}
        num = 0
        for i in range(self._num_user):
            for j in range(self._num_user):
                delay_index = self.get_delay_index(i, j)
                i_a, i_b0, i_b, i_c, i_r  = self.get_service_index(i, j)
                service_set = {i_a, i_b0, i_b, i_c, i_r}
                exist = False
                for key, value in self._delay_without_duplicate.items():
                    if value['service_set'] == service_set:
                        exist = True
                        break
                if not exist:
                    self._delay_without_duplicate[delay_index] = {'service_set': service_set, 'users':(i, j), 'new_index': num}
                    num += 1
        
    
    def _print(self):
        
        print('---instance')
        for instance in self._instances:
            print(instance)
        print('---group')
        for group in self._groups:
            print(group)
        print('---user')
        for user in self._users:
            print(user)
        
        print('关联关系：')
        print('a ', self.user_service_a)
        print('b ', self.user_service_b)
        print('c ', self.user_service_c)
        print('r ', self.user_service_r)

        for k in range(self._num_service_a):
            print(self._service_A[k])
        for k in range(self._num_service_b):
            print(self._service_B[k])
        for k in range(self._num_service_c):
            print(self._service_C[k])
        for k in range(self._num_service_r):
            print(self._service_R[k])
    
    def save_price_list(self):
        self._price_list = []
        for i in range(self._num_service):
            _type, index  = self.get_service_type_and_index(i)
            if _type == 'a':    
                self._price_list.append(self._service_A[index]._price)
            elif _type == 'b0':
                self._price_list.append(self._service_b0._price)
            elif _type == 'b':
                self._price_list.append(self._service_B[index]._price)
            elif _type == 'c':
                if len(self._service_C[index]._group) == 0:
                    self._price_list.append(self._service_C[index]._price)
                else:
                    self._price_list.append(self._service_C[index]._price * len(self._service_C[index]._group))
            elif _type == 'r':
                self._price_list.append(self._service_R[index]._price)     
        self._min_price_list = min(self._price_list)
    
    def get_service_type_and_index(self, i):
        _type = None
        index = -1
        if i < self._num_service_a:
            _type = 'a'
            index = i
        elif i == self._num_service_a:
            _type = 'b0'
            index = 0
        elif i > self._num_service_a and i < self._num_service_a + 1 + self._num_service_b:
            _type = 'b'
            index = i - self._num_service_a - 1
        elif i >= self._num_service_a + 1 + self._num_service_b and i < self._num_service_a + 1 + self._num_service_b + self._num_service_c:
            _type = 'c'
            index = i - self._num_service_a - 1 - self._num_service_b
        else:
            _type = 'r'
            index = i - self._num_service_a - 1 - self._num_service_b - self._num_service_c
        return _type, index    
    
    def get_service_index(self, user_i, user_j):   
        a_index = self._users[user_i]._service_a
        b_index = self._users[user_j]._service_b
        c_index = self._users[user_j]._service_c
        r_index = self._users[user_j]._service_r
        
        i_a = a_index
        i_b0 = self._num_service_a
        i_b = self._num_service_a + 1  + b_index
        i_c = self._num_service_a + 1  + self._num_service_b + c_index
        i_r = self._num_service_a + 1  + self._num_service_b + self._num_service_c + r_index

        return i_a, i_b0, i_b, i_c, i_r    
    
    def get_feature(self):     #没有去冗余
        service_feature = [] #到达率，服务率，当前的处理器个数，当前的排队时延，服务器单价
        for i in range(self._num_service):
            service_feature_i = []
            _type, index = self.get_service_type_and_index(i)
            if _type == 'a':    
                service_feature_i.append(self._service_A[index]._arrival_rate)
                service_feature_i.append(self._service_A[index]._service_rate)
                service_feature_i.append(self._service_A[index]._num_server)
                service_feature_i.append(self._service_A[index]._queuing_delay* 1000)
                service_feature_i.append(self._service_A[index].reduction_of_delay_when_add_a_server()* 1000)
                service_feature_i.append(self._service_A[index]._price)
            elif _type == 'b0':
                service_feature_i.append(self._service_b0._arrival_rate)
                service_feature_i.append(self._service_b0._service_rate)
                service_feature_i.append(self._service_b0._num_server)
                service_feature_i.append(self._service_b0._queuing_delay* 1000)
                service_feature_i.append(self._service_b0.reduction_of_delay_when_add_a_server()* 1000)
                service_feature_i.append(self._service_b0._price)               
            elif _type == 'b':
                service_feature_i.append(self._service_B[index]._arrival_rate)
                service_feature_i.append(self._service_B[index]._service_rate)
                service_feature_i.append(self._service_B[index]._num_server)
                service_feature_i.append(self._service_B[index]._queuing_delay* 1000)
                service_feature_i.append(self._service_B[index].reduction_of_delay_when_add_a_server()* 1000)
                service_feature_i.append(self._service_B[index]._price)
            elif _type == 'c':
                service_feature_i.append(self._service_C[index]._arrival_rate)
                service_feature_i.append(self._service_C[index]._service_rate)
                service_feature_i.append(self._service_C[index]._num_server)
                service_feature_i.append(self._service_C[index]._queuing_delay* 1000)    
                service_feature_i.append(self._service_C[index].reduction_of_delay_when_add_a_server()* 1000)
                if len(self._service_C[index]._group) == 0:
                    service_feature_i.append(self._service_C[index]._price)
                else:
                    service_feature_i.append(self._service_C[index]._price * len(self._service_C[index]._group))
            elif _type == 'r':
                service_feature_i.append(self._service_R[index]._arrival_rate)
                service_feature_i.append(self._service_R[index]._service_rate)
                service_feature_i.append(self._service_R[index]._num_server)
                service_feature_i.append(self._service_R[index]._queuing_delay* 1000)
                service_feature_i.append(self._service_R[index].reduction_of_delay_when_add_a_server()* 1000)
                service_feature_i.append(self._service_R[index]._price) 
            for i in range(len(service_feature_i)):
                service_feature_i[i] = float(service_feature_i[i])
            service_feature.append(service_feature_i)
        
        delay_feature = []
        edge_indice = [[],[]]  #0:delay 1:service
        edge_feature = []
        adj = np.zeros((self._num_service, self._num_user * self._num_user)) #service-delay
        num_delay = 0
        for i in range(self._num_user):
            for j in range(self._num_user):   
                queuing_delay = self.compute_queuing_delay(i, j)
                delay_feature.append([queuing_delay * 1000])
                             
                i_a, i_b0, i_b, i_c, i_r = self.get_service_index(i, j)
                edge_indice[1].append(i_a)
                edge_indice[1].append(i_b0)
                edge_indice[1].append(i_b)
                edge_indice[1].append(i_c)
                edge_indice[1].append(i_r)
                adj[i_a][num_delay] = 1
                adj[i_b0][num_delay] = 1
                adj[i_b][num_delay] = 1
                adj[i_c][num_delay] = 1
                adj[i_r][num_delay] = 1
                for k in range(5):
                    edge_indice[0].append(num_delay)
                    edge_feature.append([1.])
                num_delay += 1
                
        #json不支持int类型，转换为float
        for r in range(len(edge_indice)):
            for c in range(len(edge_indice[0])):
                edge_indice[r][c] = float(edge_indice[r][c])
        
        return delay_feature, service_feature, edge_feature, edge_indice, adj.tolist()  
        
    def initialize_user(self):
        self._instances = []
        self._groups = []
        self._users = []
        
        instance_id = 0
        group_id = 0
        user_id = 0
        
        #二维矩阵保存关联关系
        self._num_user = self._num_instance * self._num_group_per_instance * self._num_user_per_group
        self.user_service_a = np.zeros((self._num_user, self._num_service_a))
        self.user_service_b = np.zeros((self._num_user, self._num_service_b))
        self.user_service_c = np.zeros((self._num_user, self._num_service_c))
        self.user_service_r = np.zeros((self._num_user, self._num_service_r))
        
        for i in range(self._num_instance):
            instance = Instance(instance_id)
            for g in range(self._num_group_per_instance):
                group = ShareViewGroup(group_id, instance_id)
                for u in range(self._num_user_per_group):
                    user = User(user_id, instance_id, group_id)
                    user._arrival_rate = self._rng.integers(self._min_arrival_rate, self._max_arrival_rate)
                    user._service_a = self._rng.integers(0, self._num_service_a)
                    user._service_b = self._rng.integers(0, self._num_service_b)
                    user._service_c = self._rng.integers(0, self._num_service_c)
                    user._service_r = self._rng.integers(0, self._num_service_r)
                    '''
                    #以一定的概率跟同组相同
                    if u >0:
                        e = self._rng.random()
                        if e>0.5:
                            user._service_c = self._users[user_id-1]._service_c
                            user._service_r = self._users[user_id-1]._service_r           
                    '''
                    
                    #二维矩阵保存关联关系
                    self.user_service_a[user._id][user._service_a] = 1
                    self.user_service_b[user._id][user._service_b] = 1
                    self.user_service_c[user._id][user._service_c] = 1
                    self.user_service_r[user._id][user._service_r] = 1
                    self._users.append(user)
                    group._users.append(user_id)
                    user_id += 1
                self._groups.append(group)
                instance._groups.append(group_id)
                group_id += 1
            self._instances.append(instance)
            instance_id += 1

    def initialize_service(self):
        #传输时延
        self._tx_u_a = self._rng.integers(self._min_tx, self._max_tx, (self._num_user, self._num_service_a))
        self._tx_a_b0 = self._rng.integers(self._min_tx, self._max_tx, self._num_service_a)
        self._tx_b0_b = self._rng.integers(self._min_tx, self._max_tx, self._num_service_b)
        self._tx_b_c = self._rng.integers(self._min_tx, self._max_tx, (self._num_service_b, self._num_service_c))
        self._tx_c_r = self._rng.integers(self._min_tx, self._max_tx, (self._num_service_c, self._num_service_r))
        self._tx_r_u = self._rng.integers(self._min_tx, self._max_tx, (self._num_service_r, self._num_user))
        
        self._service_A = []
        self._service_B = []
        self._service_C = []
        self._service_R = []
        #service_A
        for a in range(self._num_service_a):
            service = Service(a, 'a')
            service._service_rate = self._rng.integers(self._min_service_rate, self._max_service_rate)
            for u in range(self._num_user):
                if self.user_service_a[u][a] == 1:
                    service._arrival_rate += self._users[u]._arrival_rate
                    service._users.append(u)
            service._price = self._rng.integers(self._min_price, self._max_price)
            self._service_A.append(service)
        #service_b0
        self._service_b0 = Service(0, 'b0')
        self._service_b0._service_rate = self._rng.integers(self._min_service_rate, self._max_service_rate)
        for u in range(self._num_user):
            self._service_b0._arrival_rate += self._users[u]._arrival_rate
            self._service_b0._users.append(u)
        self._service_b0._price = self._rng.integers(self._min_price, self._max_price)
        #service_B
        for b in range(self._num_service_b):
            service = Service(b, 'b')
            service._service_rate = self._rng.integers(self._min_service_rate, self._max_service_rate)
            service._arrival_rate = self._service_b0._arrival_rate
            for u in range(self._num_user):
                if self.user_service_b[u][b] == 1:
                    service._users.append(u)
            service._price = self._rng.integers(self._min_price, self._max_price)
            self._service_B.append(service)
        #service_C
        for c in range(self._num_service_c):
            service = Service(c, 'c')
            service._service_rate = self._rng.integers(self._min_service_rate, self._max_service_rate)
            service._arrival_rate = self._service_b0._arrival_rate * self._trigger_probability
            for u in range(self._num_user):
                if self.user_service_c[u][c] == 1:
                    service._users.append(u)
                    service._group.add(self._users[u]._group_id)
            service._price = self._rng.integers(self._min_price, self._max_price)
            self._service_C.append(service)
        #service_R
        for r in range(self._num_service_r):
            service = Service(r, 'r')
            service._service_rate = self._rng.integers(self._min_service_rate, self._max_service_rate)
            for u in range(self._num_user):
                if self.user_service_r[u][r] == 1:
                    service._users.append(u)
                    service._group.add(self._users[u]._group_id)
            group_cr = set()
            for c in range(self._num_service_c):
                for u in self._service_C[c]._users:
                    if u in service._users:
                        group_cr.add(self._users[u]._group_id)
            assert group_cr == service._group
            service._arrival_rate = len(group_cr) * self._service_b0._arrival_rate * self._trigger_probability
            service._price = self._rng.integers(self._min_price, self._max_price)
            self._service_R.append(service)
    
    def correct_initialize_user(self): #避免同一用户组的任意两个用户，关联的服务c不同，而关联的服务r相同
        for group in self._groups: #如果出现此种情况，将用户关联相同的服务c
            for u in group._users:
                u_cmp = u + 1
                while u_cmp <= group._users[-1]:
                    if self._users[u]._service_c != self._users[u_cmp]._service_c and self._users[u]._service_r == self._users[u_cmp]._service_r:
                        self.user_service_c[u_cmp][self._users[u_cmp]._service_c] = 0
                        self._users[u_cmp]._service_c = self._users[u]._service_c
                        self.user_service_c[u_cmp][self._users[u_cmp]._service_c] = 1
                    u_cmp += 1
    
    def compute_interaction_delay(self, user_i, user_j):
        #计算时延
        computation_delay = 1/self._service_A[self._users[user_i]._service_a]._service_rate +\
                            1/self._service_b0._service_rate +\
                            1/self._service_B[self._users[user_j]._service_b]._service_rate +\
                            1/self._service_C[self._users[user_j]._service_c]._service_rate +\
                            1/self._service_R[self._users[user_j]._service_r]._service_rate 
        #传输时延
        transmission_delay = self._tx_u_a[user_i][self._users[user_i]._service_a] +\
                             self._tx_a_b0[self._users[user_i]._service_a] +\
                             self._tx_b0_b[self._users[user_j]._service_b] +\
                             self._tx_b_c[self._users[user_j]._service_b][self._users[user_j]._service_c] +\
                             self._tx_c_r[self._users[user_j]._service_c][self._users[user_j]._service_r] +\
                             self._tx_r_u[self._users[user_j]._service_r][user_j]
        #排队时延
        queuing_delay = self.compute_queuing_delay(user_i, user_j)
        delay = computation_delay + transmission_delay/1000 + queuing_delay
        return delay
    
    def compute_queuing_delay(self, user_i, user_j):
        queuing_delay = self._service_A[self._users[user_i]._service_a]._queuing_delay +\
                        self._service_b0._queuing_delay +\
                        self._service_B[self._users[user_j]._service_b]._queuing_delay +\
                        self._service_C[self._users[user_j]._service_c]._queuing_delay +\
                        self._service_R[self._users[user_j]._service_r]._queuing_delay 
        return queuing_delay
    
    def compute_cost(self):
        cost = 0
        for k in range(self._num_service_a):
            cost += self._service_A[k]._num_server * self._service_A[k]._price
        cost += self._service_b0._num_server * self._service_b0._price
        for k in range(self._num_service_b):
            cost += self._service_B[k]._num_server * self._service_B[k]._price
        for k in range(self._num_service_c):
            if len(self._service_C[k]._group) == 0:
                cost += self._service_C[k]._num_server * self._service_C[k]._price
            else:
                cost += self._service_C[k]._num_server * self._service_C[k]._price * len(self._service_C[k]._group)
        for k in range(self._num_service_r):
            cost += self._service_R[k]._num_server * self._service_R[k]._price
        return cost
    
    def get_max_interaction_delay(self):
        self._interaction_delay_dict = {}
        for i in range(self._num_user):
            for j in range(self._num_user):
                self._interaction_delay_dict[(i,j)] = self.compute_interaction_delay(i, j)
        '''
        interaction_delay_tuple = zip(self._interaction_delay_dict.values(), self._interaction_delay_dict.keys())
        interaction_delay_list = sorted(interaction_delay_tuple)
        max_interaction_delay = interaction_delay_list[-1]
        return max_interaction_delay[0], max_interaction_delay[1][0], max_interaction_delay[1][1]
        '''
        max_user_pair = max(self._interaction_delay_dict, key=lambda k: self._interaction_delay_dict[k])
        return self._interaction_delay_dict[max_user_pair], max_user_pair[0], max_user_pair[1]   
    
    def get_max_two_queuing_delay_in_interaction_delay(self):  #去冗余
        self._queuing_delay_dict = {}
        for key, value in self._delay_without_duplicate.items():
            user_i = value['users'][0]
            user_j = value['users'][1]
            self._queuing_delay_dict[(user_i, user_j)] = self.compute_queuing_delay(user_i, user_j)
    
        delay_tuple = zip(self._queuing_delay_dict.values(), self._queuing_delay_dict.keys())
        delay_list = sorted(delay_tuple)
        
        max_delay = delay_list[-1]         #i:max_delay[1][0], j:max_delay[1][1]
        _2nd_max_delay = delay_list[-2]
        return max_delay, _2nd_max_delay
    
    def get_delay_index(self, user_i , user_j):
        index = 0
        for i in range(self._num_user):
            for j in range(self._num_user):
                if user_i == i and user_j == j:
                    return index
                else:
                    index += 1    
        return -1    
    
    def get_topk_queuing_delay_in_interaction_delay(self, topk):  #去冗余
        self._queuing_delay_dict = {}
        for key, value in self._delay_without_duplicate.items():
            user_i = value['users'][0]
            user_j = value['users'][1]
            self._queuing_delay_dict[(user_i, user_j)] = self.compute_queuing_delay(user_i, user_j)
    
        delay_tuple = zip(self._queuing_delay_dict.values(), self._queuing_delay_dict.keys())
        delay_list = sorted(delay_tuple)
        
        index_list = []
        k = -1
        for i in range(topk):
            delay_index = self.get_delay_index(delay_list[k][1][0], delay_list[k][1][1])
            index_list.append(delay_index)
            k -= 1
        return index_list
            
    
    def get_max_queuing_delay_in_interaction_delay_old(self): ############没有去冗余
        self._queuing_delay_dict = {}
        for i in range(self._num_user):
            for j in range(self._num_user):
                self._queuing_delay_dict[(i,j)] = self.compute_queuing_delay(i, j)
        max_user_pair = max(self._queuing_delay_dict, key=lambda k: self._queuing_delay_dict[k])
        return self._queuing_delay_dict[max_user_pair], max_user_pair[0], max_user_pair[1]
    
    def get_max_queuing_delay_in_interaction_delay(self):  #去冗余
        self._queuing_delay_dict = {}
        for key, value in self._delay_without_duplicate.items():
            user_i = value['users'][0]
            user_j = value['users'][1]
            self._queuing_delay_dict[(user_i, user_j)] = self.compute_queuing_delay(user_i, user_j)
        max_user_pair = max(self._queuing_delay_dict, key=lambda k: self._queuing_delay_dict[k])
        return self._queuing_delay_dict[max_user_pair], max_user_pair[0], max_user_pair[1]    
    
    def get_total_interaction_delay(self):
        total_interaction_delay = 0.
        for i in range(self._num_user):
            for j in range(self._num_user):
                total_interaction_delay += self.compute_interaction_delay(i, j)
        return total_interaction_delay
    
    def get_total_queuing_delay(self):
        total_queuing_delay = 0.
        for i in range(self._num_user):
            for j in range(self._num_user):
                total_queuing_delay += self.compute_queuing_delay(i, j)
        return total_queuing_delay
    
    def initialize_num_server(self):
        for k in range(self._num_service_a):
            self._service_A[k].initialize_num_server()
        self._service_b0.initialize_num_server()
        for k in range(self._num_service_b):
            self._service_B[k].initialize_num_server()
        for k in range(self._num_service_c):
            self._service_C[k].initialize_num_server()
        for k in range(self._num_service_r):
            self._service_R[k].initialize_num_server()
        cost = self.compute_cost()
        self._cost_budget = cost + self._budget_addition
        self.save_initial_num_server()
        max_delay, user_i, user_j = self.get_max_queuing_delay_in_interaction_delay()
        self.initial_max_delay = max_delay * 1000        
    
    def re_initialize_num_server(self):
        for i in range(self._num_service):
            _type, index = self.get_service_type_and_index(i)
            if _type == 'a':
                self._service_A[index].update_num_server(self._initial_num_server[i])
            elif _type == 'b0':
                self._service_b0.update_num_server(self._initial_num_server[i])              
            elif _type == 'b':
                self._service_B[index].update_num_server(self._initial_num_server[i])
            elif _type == 'c':
                self._service_C[index].update_num_server(self._initial_num_server[i])
            else:
                assert _type == 'r'
                self._service_R[index].update_num_server(self._initial_num_server[i])   
        #cost = self.compute_cost()
        #print('=====================reset cost={}, budget={}, '.format(cost, self._cost_budget))
    
    def save_initial_num_server(self):
        self._initial_num_server = []
        for i in range(self._num_service):
            _type, index = self.get_service_type_and_index(i)
            n = 0
            if _type == 'a':
                n = self._service_A[index]._num_server
            elif _type == 'b0':
                n = self._service_b0._num_server
            elif _type == 'b':
                n = self._service_B[index]._num_server
            elif _type == 'c':
                n = self._service_C[index]._num_server
            else:
                assert _type == 'r'
                n = self._service_R[index]._num_server
            self._initial_num_server.append(n)
        return    

    def valid_action_mask(self):
        cost = self.compute_cost()
        mask = []
        for i in range(self._num_service):
            if self._price_list[i] > self._cost_budget - cost:
                mask.append(False)
            else:
                mask.append(True)
        return mask
    
    def observe(self):  #去冗余
        delay_feature, service_feature, edge_feature, edge_indice, adj = self.get_feature_without_duplicate()
        return [delay_feature, service_feature, edge_feature, edge_indice, adj]

    def act(self, action):
        #cost1 = self.compute_cost()
        max_delay_before, user_i, user_j = self.get_max_queuing_delay_in_interaction_delay()
        avg_queuing_delay_before = self.get_total_queuing_delay()/(self._num_user * self._num_user)
        _type, index = self.get_service_type_and_index(action)

        if _type == 'a':
            self._service_A[index].update_num_server(self._service_A[index]._num_server + 1)
        elif _type == 'b0':
            self._service_b0.update_num_server(self._service_b0._num_server + 1)
        elif _type == 'b':
            self._service_B[index].update_num_server(self._service_B[index]._num_server + 1)
        elif _type == 'c':
            self._service_C[index].update_num_server(self._service_C[index]._num_server + 1)
        elif _type == 'r':
            self._service_R[index].update_num_server(self._service_R[index]._num_server + 1)        
        
        max_delay_after, user_i, user_j = self.get_max_queuing_delay_in_interaction_delay()
        max_delay_reduction_ratio = (max_delay_before - max_delay_after) / max_delay_before
        avg_queuing_delay_after = self.get_total_queuing_delay()/(self._num_user * self._num_user)
        avg_delay_reduction_ratio = (avg_queuing_delay_before - avg_queuing_delay_after) / avg_queuing_delay_before
        
        reward = max_delay_reduction_ratio#/2 + avg_delay_reduction_ratio/2
        
        done = True
        cost2 = self.compute_cost()
        if cost2 < self._cost_budget and self._cost_budget - cost2 >= self._min_price_list:
            done = False        
        if done:
            assert cost2 <= self._cost_budget
            print()
            print('=======done cost={}, budget={}, [max delay]={}'.format(cost2, self._cost_budget, round(max_delay_after* 1000, 2)))   

        return reward, done
    
    
    def get_feature_without_duplicate(self):     #去冗余
        service_feature = [] #到达率，服务率，当前的处理器个数，当前的排队时延，服务器单价
        for i in range(self._num_service):
            service_feature_i = []
            _type, index = self.get_service_type_and_index(i)
            if _type == 'a':    
                service_feature_i.append(self._service_A[index]._arrival_rate)
                service_feature_i.append(self._service_A[index]._service_rate)
                service_feature_i.append(self._service_A[index]._num_server)
                service_feature_i.append(self._service_A[index]._queuing_delay* 1000)
                service_feature_i.append(self._service_A[index].reduction_of_delay_when_add_a_server()* 1000)
                service_feature_i.append(self._service_A[index]._price)
            elif _type == 'b0':
                service_feature_i.append(self._service_b0._arrival_rate)
                service_feature_i.append(self._service_b0._service_rate)
                service_feature_i.append(self._service_b0._num_server)
                service_feature_i.append(self._service_b0._queuing_delay* 1000)
                service_feature_i.append(self._service_b0.reduction_of_delay_when_add_a_server()* 1000)
                service_feature_i.append(self._service_b0._price)               
            elif _type == 'b':
                service_feature_i.append(self._service_B[index]._arrival_rate)
                service_feature_i.append(self._service_B[index]._service_rate)
                service_feature_i.append(self._service_B[index]._num_server)
                service_feature_i.append(self._service_B[index]._queuing_delay* 1000)
                service_feature_i.append(self._service_B[index].reduction_of_delay_when_add_a_server()* 1000)
                service_feature_i.append(self._service_B[index]._price)
            elif _type == 'c':
                service_feature_i.append(self._service_C[index]._arrival_rate)
                service_feature_i.append(self._service_C[index]._service_rate)
                service_feature_i.append(self._service_C[index]._num_server)
                service_feature_i.append(self._service_C[index]._queuing_delay* 1000)    
                service_feature_i.append(self._service_C[index].reduction_of_delay_when_add_a_server()* 1000)
                if len(self._service_C[index]._group) == 0:
                    service_feature_i.append(self._service_C[index]._price)
                else:
                    service_feature_i.append(self._service_C[index]._price * len(self._service_C[index]._group))
            elif _type == 'r':
                service_feature_i.append(self._service_R[index]._arrival_rate)
                service_feature_i.append(self._service_R[index]._service_rate)
                service_feature_i.append(self._service_R[index]._num_server)
                service_feature_i.append(self._service_R[index]._queuing_delay* 1000)
                service_feature_i.append(self._service_R[index].reduction_of_delay_when_add_a_server()* 1000)
                service_feature_i.append(self._service_R[index]._price) 
            for i in range(len(service_feature_i)):
                service_feature_i[i] = float(service_feature_i[i])
            service_feature.append(service_feature_i)
        
        delay_feature = []
        edge_indice = [[],[]]  #0:delay 1:service
        edge_feature = []
        adj = np.zeros((self._num_service, len(self._delay_without_duplicate))) #service-delay
        num_delay = 0
        for key, value in self._delay_without_duplicate.items():
            assert num_delay == value['new_index']
            user_i = value['users'][0]
            user_j = value['users'][1]        

            queuing_delay = self.compute_queuing_delay(user_i, user_j)
            delay_feature.append([queuing_delay * 1000])
                         
            i_a, i_b0, i_b, i_c, i_r = self.get_service_index(user_i, user_j)
            edge_indice[1].append(i_a)
            edge_indice[1].append(i_b0)
            edge_indice[1].append(i_b)
            edge_indice[1].append(i_c)
            edge_indice[1].append(i_r)
            adj[i_a][num_delay] = 1
            adj[i_b0][num_delay] = 1
            adj[i_b][num_delay] = 1
            adj[i_c][num_delay] = 1
            adj[i_r][num_delay] = 1
            for k in range(5):
                edge_indice[0].append(num_delay)
                edge_feature.append([1.])
            num_delay += 1
                
        #json不支持int类型，转换为float
        for r in range(len(edge_indice)):
            for c in range(len(edge_indice[0])):
                edge_indice[r][c] = float(edge_indice[r][c])
        
        return delay_feature, service_feature, edge_feature, edge_indice, adj.tolist()      

    def get_topk_feature(self, topk_delay):
        
        service_feature = [] #到达率，服务率，当前的处理器个数，当前的排队时延，服务器单价
        service_set = set()   
        for i in range(len(topk_delay)):
            for s in self._delay_without_duplicate[topk_delay[i]]['service_set']:
                if s not in service_set:
                    service_set.add(s)
        service_set = list(service_set)
        service_set.sort()
        for i in range(len(service_set)):  #按下标从小到大排序
            service_feature_i = []
            _type, index = self.get_service_type_and_index(service_set[i])
            if _type == 'a':    
                service_feature_i.append(self._service_A[index]._arrival_rate)
                service_feature_i.append(self._service_A[index]._service_rate)
                service_feature_i.append(self._service_A[index]._num_server)
                service_feature_i.append(self._service_A[index]._queuing_delay* 1000)
                #service_feature_i.append(self._service_A[index].reduction_of_delay_when_add_a_server()* 1000)
                service_feature_i.append(self._service_A[index]._price)
                #service_feature_i.append(self._budget_addition)
            elif _type == 'b0':
                service_feature_i.append(self._service_b0._arrival_rate)
                service_feature_i.append(self._service_b0._service_rate)
                service_feature_i.append(self._service_b0._num_server)
                service_feature_i.append(self._service_b0._queuing_delay* 1000)
                #service_feature_i.append(self._service_b0.reduction_of_delay_when_add_a_server()* 1000)
                service_feature_i.append(self._service_b0._price)    
                #service_feature_i.append(self._budget_addition)
            elif _type == 'b':
                service_feature_i.append(self._service_B[index]._arrival_rate)
                service_feature_i.append(self._service_B[index]._service_rate)
                service_feature_i.append(self._service_B[index]._num_server)
                service_feature_i.append(self._service_B[index]._queuing_delay* 1000)
                #service_feature_i.append(self._service_B[index].reduction_of_delay_when_add_a_server()* 1000)
                service_feature_i.append(self._service_B[index]._price)
                #service_feature_i.append(self._budget_addition)
            elif _type == 'c':
                service_feature_i.append(self._service_C[index]._arrival_rate)
                service_feature_i.append(self._service_C[index]._service_rate)
                service_feature_i.append(self._service_C[index]._num_server)
                service_feature_i.append(self._service_C[index]._queuing_delay* 1000)    
                #service_feature_i.append(self._service_C[index].reduction_of_delay_when_add_a_server()* 1000)
                if len(self._service_C[index]._group) == 0:
                    service_feature_i.append(self._service_C[index]._price)
                else:
                    service_feature_i.append(self._service_C[index]._price * len(self._service_C[index]._group))
                #service_feature_i.append(self._budget_addition)
            elif _type == 'r':
                service_feature_i.append(self._service_R[index]._arrival_rate)
                service_feature_i.append(self._service_R[index]._service_rate)
                service_feature_i.append(self._service_R[index]._num_server)
                service_feature_i.append(self._service_R[index]._queuing_delay* 1000)
                #service_feature_i.append(self._service_R[index].reduction_of_delay_when_add_a_server()* 1000)
                service_feature_i.append(self._service_R[index]._price) 
                #service_feature_i.append(self._budget_addition)
                
            for j in range(len(service_feature_i)):  #json不支持int类型，转换为float
                service_feature_i[j] = float(service_feature_i[j])
            service_feature.append(service_feature_i)
        
        delay_feature = []   #按时延从大到小排序
        edge_indice = [[],[]]  #0:delay 1:service
        edge_feature = []
        adj = np.zeros((len(service_feature), len(topk_delay))) #service-delay
        for i in range(len(topk_delay)):  # {'service_set': service_set, 'users':(i, j), 'new_index': num}
            
            user_i = self._delay_without_duplicate[topk_delay[i]]['users'][0]
            user_j = self._delay_without_duplicate[topk_delay[i]]['users'][1]   
            queuing_delay = self.compute_queuing_delay(user_i, user_j)
            delay_feature.append([queuing_delay * 1000])
                         
            i_a, i_b0, i_b, i_c, i_r = self.get_service_index(user_i, user_j)
            edge_indice[1].append(service_set.index(i_a))
            edge_indice[1].append(service_set.index(i_b0))
            edge_indice[1].append(service_set.index(i_b))
            edge_indice[1].append(service_set.index(i_c))
            edge_indice[1].append(service_set.index(i_r))
            adj[service_set.index(i_a)][i] = 1
            adj[service_set.index(i_b0)][i] = 1
            adj[service_set.index(i_b)][i] = 1
            adj[service_set.index(i_c)][i] = 1
            adj[service_set.index(i_r)][i] = 1
            for k in range(5):
                edge_indice[0].append(i)
                edge_feature.append([1.])              
        
        for r in range(len(edge_indice)):  #json不支持int类型，转换为float
            for c in range(len(edge_indice[0])):
                edge_indice[r][c] = float(edge_indice[r][c])
        
        return [ delay_feature, service_feature, edge_feature, edge_indice, adj.tolist() ]  
