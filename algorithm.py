import numpy as np
import sys
from scipy.misc import derivative
import time
import matplotlib.pyplot as plt
import pulp
import cplex
from agent import *
import simplejson
from models import *
import torch

class Algorithm:
    def __init__(self, env, server_each_allocation=1):
        self._env = env
        self._max_delay = 0  # queuing delay
        self._delay_reduction = 0  # min total

        """
        2022-7-21
            每次添加服务器的个数，目前只针对min_max_greedy算法，但这个参数也适用于其他算法（其他算法未改动，默认为1）
        """
        self._server_each_allocation = server_each_allocation
    
    def get_running_time(self):
        self._end_time = time.time()
        self._running_time = (self._end_time - self._start_time)*1000
    
    def set_num_server(self):
        return
    
    def get_delay_before(self):#min total
        self._total_interaction_delay_before = self._env.get_total_interaction_delay()
        self._total_queuing_delay_before = self._env.get_total_queuing_delay()
    
    def get_result_after(self):#min total
        self._total_interaction_delay_after = self._env.get_total_interaction_delay()
        assert abs(self._delay_reduction - (self._total_interaction_delay_before - self._total_interaction_delay_after)) < 1e-5
        print('total_delay_reduction ',self._delay_reduction)
        self._total_queuing_delay_after = self._env.get_total_queuing_delay()
        self._queuing_delay_reduction_ratio = (self._total_queuing_delay_before - self._total_queuing_delay_after) / self._total_queuing_delay_before
        print('queuing_delay_reduction_ratio ',self._queuing_delay_reduction_ratio)
        cost = self._env.compute_cost()
        print('cost ', cost)           

    def get_initial_max_queuing_delay(self):#min max
        self._cost = self._env.compute_cost()
        max_delay, user_i, user_j = self._env.get_max_queuing_delay_in_interaction_delay()
        self._max_queuing_delay_initial = max_delay
        print('initial max queuing delay: [',self._max_queuing_delay_initial,'] cost ', self._cost,' budget ',self._env._cost_budget)
    
    def get_min_max_result(self):#min max
        cost = self._env.compute_cost()
        max_delay, user_i, user_j = self._env.get_max_queuing_delay_in_interaction_delay()
        self._max_delay = max_delay        
        print('final max queuing delay [',self._max_delay,'] cost ', self._cost,' budget ',self._env._cost_budget)
        assert cost <= self._env._cost_budget
        
    def get_result_dict(self):
        self.result_dict = {'max_delay':self._max_delay, 'running_time':self._running_time}
        return self.result_dict

IF_DEBUG = True

#min max （只优化排队时延）
class min_max_greedy(Algorithm):
    def __init__(self, env, server_each_allocation=1):
        Algorithm.__init__(self, env, server_each_allocation)
    
    def get_max_utility(self, services):
        services = sorted(services, key=lambda services : services['reduction'] / services['price'],reverse=True)
        max_utility = None

        # 从 self._server_each_allocation 开始尝试分配，如果太多了（预算不够了），就减少一个
        server_count = self._server_each_allocation

        for k in range(len(services)):
            server_count = self._server_each_allocation
            while server_count > 0:
                if services[k]['price'] * server_count >  self._env._cost_budget - self._cost:      # 剩余的预算足够分配 _server_each_allocation 台服务器
                    server_count -= 1
                else:
                    max_utility = services[k]['service']
                    break
            if max_utility is not  None:
                break


        # while server_count > 0:
        #     for k in range(len(services)):
        #         if services[k]['price'] * server_count >  self._env._cost_budget - self._cost:      # 剩余的预算足够分配 _server_each_allocation 台服务器
        #             continue
        #         else:
        #             max_utility = services[k]['service']
        #             break
        #     if max_utility is not None:
        #         break
        #     server_count -= 1

        return max_utility, server_count
    
    def set_num_server(self):
        self._start_time = time.time()
        self.get_initial_max_queuing_delay()
        while self._cost < self._env._cost_budget:
            max_delay, user_i, user_j = self._env.get_max_queuing_delay_in_interaction_delay()
            services = []
            services.append({'service':'a', 'reduction': self._env._service_A[self._env._users[user_i]._service_a].reduction_of_delay_when_add_a_server(), 'price':self._env._service_A[self._env._users[user_i]._service_a]._price})
            services.append({'service':'b0', 'reduction': self._env._service_b0.reduction_of_delay_when_add_a_server(), 'price':self._env._service_b0._price})
            services.append({'service':'b', 'reduction': self._env._service_B[self._env._users[user_j]._service_b].reduction_of_delay_when_add_a_server(), 'price':self._env._service_B[self._env._users[user_j]._service_b]._price})
            services.append({'service':'c', 'reduction': self._env._service_C[self._env._users[user_j]._service_c].reduction_of_delay_when_add_a_server(),
                             'price':self._env._service_C[self._env._users[user_j]._service_c]._price * len(self._env._service_C[self._env._users[user_j]._service_c]._group),
                             'price_single':self._env._service_C[self._env._users[user_j]._service_c]._price})
            services.append({'service':'r', 'reduction': self._env._service_R[self._env._users[user_j]._service_r].reduction_of_delay_when_add_a_server(), 'price':self._env._service_R[self._env._users[user_j]._service_r]._price})

            if IF_DEBUG:
                a, b0, b, c, r = self._env.get_service_index(user_i, user_j)
                indices = (a, b0, b, c, r)
                print("service indices: {}".format(indices))
                print("max_delay: {}, users: ({}, {})".format(max_delay, user_i, user_j))
                print("services: {}".format(services))

            #print(max_utility)
            # if max_utility:
            #     if max_utility == 'a':
            #         self._env._service_A[self._env._users[user_i]._service_a].update_num_server(self._env._service_A[self._env._users[user_i]._service_a]._num_server + 1)
            #     elif max_utility == 'b0':
            #         self._env._service_b0.update_num_server(self._env._service_b0._num_server + 1)
            #     elif max_utility == 'b':
            #         self._env._service_B[self._env._users[user_j]._service_b].update_num_server(self._env._service_B[self._env._users[user_j]._service_b]._num_server + 1)
            #     elif max_utility == 'c':
            #         self._env._service_C[self._env._users[user_j]._service_c].update_num_server(self._env._service_C[self._env._users[user_j]._service_c]._num_server + 1)
            #     elif max_utility == 'r':
            #         self._env._service_R[self._env._users[user_j]._service_r].update_num_server(self._env._service_R[self._env._users[user_j]._service_r]._num_server + 1)
            # else:
            #     break

            """
                改成一次增加多台台服务器（1-self._server_each_allocation）
            """
            max_utility, server_to_add = self.get_max_utility(services)

            if IF_DEBUG:
                print("max_utility: {}, add server: {}".format(max_utility, server_to_add))

            if max_utility:
                if max_utility == 'a':
                    self._env._service_A[self._env._users[user_i]._service_a].update_num_server(
                        self._env._service_A[self._env._users[user_i]._service_a]._num_server + server_to_add)
                elif max_utility == 'b0':
                    self._env._service_b0.update_num_server(self._env._service_b0._num_server + server_to_add)
                elif max_utility == 'b':
                    self._env._service_B[self._env._users[user_j]._service_b].update_num_server(
                        self._env._service_B[self._env._users[user_j]._service_b]._num_server + server_to_add)
                elif max_utility == 'c':
                    self._env._service_C[self._env._users[user_j]._service_c].update_num_server(
                        self._env._service_C[self._env._users[user_j]._service_c]._num_server + server_to_add)
                elif max_utility == 'r':
                    self._env._service_R[self._env._users[user_j]._service_r].update_num_server(
                        self._env._service_R[self._env._users[user_j]._service_r]._num_server + server_to_add)
            else:
                break
            self._cost = self._env.compute_cost()

            if IF_DEBUG:
                print("cost: {}, cost_budget: {}\n".format(self._cost, self._env._cost_budget))

        
        self.get_min_max_result()
        self.get_running_time()
        return

class min_max_delay_only(min_max_greedy):
    def __init__(self, env):
        min_max_greedy.__init__(self, env)
    
    def get_max_utility(self, services):
        services = sorted(services, key=lambda services : services['reduction'],reverse=True)
        max_utility = None
        for k in range(len(services)):
            if services[k]['price'] >  self._env._cost_budget - self._cost:
                continue
            else:
                max_utility = services[k]['service']
                break
        return max_utility

class min_max_random(min_max_greedy):
    def __init__(self, env):
        min_max_greedy.__init__(self, env)
    
    def get_max_utility(self, services):
        self._env._rng.shuffle(services)
        max_utility = None
        for k in range(len(services)):
            if services[k]['price'] >  self._env._cost_budget - self._cost:
                continue
            else:
                max_utility = services[k]['service']
                break
        return max_utility

class min_max_surrogate_relaxation(Algorithm):
    def __init__(self, env):
        Algorithm.__init__(self, env)
        
        self._upper_bound = sys.maxsize
        self._lower_bound = -sys.maxsize
        
        self._upper_bound_list = []
        self._lower_bound_list = []   
        
        self.best_max_delay = sys.maxsize
    
    def get_max_utility(self, services):#min max greedy
        services = sorted(services, key=lambda services : services['reduction']/services['price'],reverse=True)
        max_utility = None
        for k in range(len(services)):
            if services[k]['price'] >  self._env._cost_budget - self._cost:
                continue
            else:
                max_utility = services[k]['service']
                break
        return max_utility
    
    def set_num_server_min_max_greedy(self):
        #print('---min max greedy---')
        self.get_initial_max_queuing_delay()
        while self._cost < self._env._cost_budget:
            max_delay, user_i, user_j = self._env.get_max_queuing_delay_in_interaction_delay()
            #print('max_delay', max_delay)
            services = []
            services.append({'service':'a', 'reduction': self._env._service_A[self._env._users[user_i]._service_a].reduction_of_delay_when_add_a_server(), 'price':self._env._service_A[self._env._users[user_i]._service_a]._price})
            services.append({'service':'b0', 'reduction': self._env._service_b0.reduction_of_delay_when_add_a_server(), 'price':self._env._service_b0._price})
            services.append({'service':'b', 'reduction': self._env._service_B[self._env._users[user_j]._service_b].reduction_of_delay_when_add_a_server(), 'price':self._env._service_B[self._env._users[user_j]._service_b]._price})
            services.append({'service':'c', 'reduction': self._env._service_C[self._env._users[user_j]._service_c].reduction_of_delay_when_add_a_server(), 'price':self._env._service_C[self._env._users[user_j]._service_c]._price * len(self._env._service_C[self._env._users[user_j]._service_c]._group)})
            services.append({'service':'r', 'reduction': self._env._service_R[self._env._users[user_j]._service_r].reduction_of_delay_when_add_a_server(), 'price':self._env._service_R[self._env._users[user_j]._service_r]._price})
            max_utility = self.get_max_utility(services)
            #print(max_utility)
            if max_utility:
                if max_utility == 'a':
                    self._env._service_A[self._env._users[user_i]._service_a].update_num_server(self._env._service_A[self._env._users[user_i]._service_a]._num_server + 1)
                elif max_utility == 'b0':
                    self._env._service_b0.update_num_server(self._env._service_b0._num_server + 1)
                elif max_utility == 'b':
                    self._env._service_B[self._env._users[user_j]._service_b].update_num_server(self._env._service_B[self._env._users[user_j]._service_b]._num_server + 1)
                elif max_utility == 'c':
                    self._env._service_C[self._env._users[user_j]._service_c].update_num_server(self._env._service_C[self._env._users[user_j]._service_c]._num_server + 1)
                elif max_utility == 'r':
                    self._env._service_R[self._env._users[user_j]._service_r].update_num_server(self._env._service_R[self._env._users[user_j]._service_r]._num_server + 1)
            else:
                break
            self._cost = self._env.compute_cost()
        self.get_min_max_result()
        return             
    
    def save_constraint_weight(self, lambda_k):
        #self.service_weight = {}
        self.constraint_weight = {}    
        w = 0
        for constraint in self._constraints:
            
            weight = lambda_k
            if w == len(self._constraints) - 1:
                weight = 1-lambda_k
            
            if constraint['users'] in self.constraint_weight.keys():
                self.constraint_weight[constraint['users']] += constraint['lambda'] * weight
            else:
                self.constraint_weight[constraint['users']] = constraint['lambda'] * weight               
            
            w += 1  
        
        sum_w = 0
        for v in self.constraint_weight.values():
            sum_w += v
        assert abs(sum_w - 1) < 1e-5
        return
    
    def get_constraint_weight(self):
        #print(self.constraint_weight)
        self.constraint_weight_list = []
        for i in range(self._env._num_user):
            for j in range(self._env._num_user):
                #print('(i ', i, ' j ', j,') : ' ,end="")
                if (i,j) in self.constraint_weight.keys():
                    self.constraint_weight_list.append(self.constraint_weight[(i, j)])
                    #print(self.constraint_weight[(i, j)])
                else:
                    self.constraint_weight_list.append(0.)
                    #print(0)
        #print(self.constraint_weight_list)
        
    
    def get_max_utility_min_sum(self, services):       
        value_list = []
        for v in services.values():
            value_list.append(v)
        value_list = sorted(value_list, key=lambda value_list : value_list['reduction']/value_list['price'],reverse=True)

        max_utility = None
        for k in range(len(value_list)):
            if value_list[k]['price'] >  self._env._cost_budget - self._cost:
                continue
            else:
                max_utility = value_list[k]
                break
        return max_utility    
    
    def compute_min_sum_allocation_solution(self, lambda_k):
        self._env.re_initialize_num_server()
        #print('-------min sum----------')
        self._cost = self._env.compute_cost()
        while self._cost < self._env._cost_budget:
            services = {}
            w = 0
            for constraint in self._constraints:
                weight = lambda_k
                if w == len(self._constraints) - 1:
                    weight = 1-lambda_k       # ?
                
                if ('a',constraint['a']) in services.keys():
                    services[('a',constraint['a'])]['reduction'] += self._env._service_A[constraint['a']].reduction_of_delay_when_add_a_server() * constraint['lambda'] * weight
                else:
                    services[('a',constraint['a'])] = {'type':'a', 'index':constraint['a'], 
                                     'reduction': self._env._service_A[constraint['a']].reduction_of_delay_when_add_a_server() * constraint['lambda'] * weight,
                                     'price':self._env._service_A[constraint['a']]._price}

                if ('b0', 0) in services.keys():
                    services[('b0', 0)]['reduction'] += self._env._service_b0.reduction_of_delay_when_add_a_server() * constraint['lambda'] * weight
                else:
                    services[('b0', 0)] = {'type':'b0', 'index':0, 
                                    'reduction': self._env._service_b0.reduction_of_delay_when_add_a_server() * constraint['lambda'] * weight,
                                    'price':self._env._service_b0._price}   
                    
                if ('b',constraint['b']) in services.keys():
                    services[('b',constraint['b'])]['reduction'] += self._env._service_B[constraint['b']].reduction_of_delay_when_add_a_server() * constraint['lambda'] * weight
                else:
                    services[('b',constraint['b'])] =  {'type':'b', 'index':constraint['b'], 
                                    'reduction': self._env._service_B[constraint['b']].reduction_of_delay_when_add_a_server() * constraint['lambda'] * weight,
                                    'price':self._env._service_B[constraint['b']]._price}              
                    
                if ('c', constraint['c']) in services.keys():
                    services[('c', constraint['c'])]['reduction'] += self._env._service_C[constraint['c']].reduction_of_delay_when_add_a_server() * constraint['lambda'] * weight
                else:
                    services[('c', constraint['c'])] = {'type':'c', 'index':constraint['c'], 
                                    'reduction': self._env._service_C[constraint['c']].reduction_of_delay_when_add_a_server() * constraint['lambda'] * weight,
                                    'price':self._env._service_C[constraint['c']]._price  * len(self._env._service_C[constraint['c']]._group)}
                
                if ('r', constraint['r']) in services.keys():
                    services[('r', constraint['r'])]['reduction'] += self._env._service_R[constraint['r']].reduction_of_delay_when_add_a_server() * constraint['lambda'] * weight
                else:
                    services[('r', constraint['r'])] = {'type':'r', 'index':constraint['r'], 
                                    'reduction': self._env._service_R[constraint['r']].reduction_of_delay_when_add_a_server() * constraint['lambda'] * weight,
                                    'price':self._env._service_R[constraint['r']]._price}                    
                w += 1       
            max_utility = self.get_max_utility_min_sum(services)
            #print('==========add server for: ',max_utility)
            if max_utility:
                if max_utility['type'] == 'a':
                    self._env._service_A[max_utility['index']].update_num_server(self._env._service_A[max_utility['index']]._num_server + 1)
                elif max_utility['type'] == 'b0':
                    self._env._service_b0.update_num_server(self._env._service_b0._num_server + 1)
                elif max_utility['type'] == 'b':
                    self._env._service_B[max_utility['index']].update_num_server(self._env._service_B[max_utility['index']]._num_server + 1)
                elif max_utility['type'] == 'c':
                    self._env._service_C[max_utility['index']].update_num_server(self._env._service_C[max_utility['index']]._num_server + 1)
                elif max_utility['type'] == 'r':
                    self._env._service_R[max_utility['index']].update_num_server(self._env._service_R[max_utility['index']]._num_server + 1)
            else:
                break
            self._cost = self._env.compute_cost()   

        cost = self._env.compute_cost()
        max_delay, user_i, user_j = self._env.get_max_queuing_delay_in_interaction_delay()
        self._max_delay = max_delay        
        assert cost <= self._env._cost_budget    
        
        f_lambda = 0
        w = 0
        for constraint in self._constraints:
            weight = lambda_k
            if w == len(self._constraints) - 1:
                weight = 1-lambda_k
            f_lambda += self._env.compute_queuing_delay(constraint['users'][0], constraint['users'][1]) * constraint['lambda'] * weight
            w+=1
        
        return f_lambda

    def line_search(self):
        #print('--------line_search-----------')
        epsilon = 0.05
        delta = 0.05
        lambda_low = 0.
        lambda_high = 1.
        k = 1
        f_lambda = 0
        lambda_k = 0
        while True:
            #print('lambda [{}, {}]  f_lambda = {}'.format(lambda_low, lambda_high, f_lambda))
            if lambda_high - lambda_low < epsilon:
                break
            lambda_k = lambda_low + (lambda_high - lambda_low)/2

            f_lambda_delta = self.compute_min_sum_allocation_solution(lambda_k + delta)
            f_lambda = self.compute_min_sum_allocation_solution(lambda_k)

            self._lower_bound_list.append( f_lambda )
            self._upper_bound_list.append( self._max_delay )
            
            if self.best_max_delay > self._max_delay:
                self.best_max_delay = self._max_delay
                self.save_constraint_weight(lambda_k)
                   
            if (f_lambda_delta - f_lambda)/delta <= 0:
                lambda_high = lambda_k
            else:
                lambda_low = lambda_k

            k += 1
        w = 0
        for i in range(len(self._constraints)):
            if w == len(self._constraints) - 1:
                self._constraints[i]['lambda'] *= (1-lambda_k)      
            else:
                self._constraints[i]['lambda'] *= lambda_k      
            w+=1
        return f_lambda
    
    def random_select_two_constraint(self):
        self._constraints =[] #保存约束（一个客户端对的排队时延）的权重、其中包含的服务         
        for r in range(2):
            random_i = self._env._rng.choice(self._env._num_user * self._env._num_user)
            k = 0
            add = False
            for i in range(self._env._num_user):
                for j in range(self._env._num_user):
                    if k == random_i:
                        self._constraints.append(  {  'users':(i,j), 'lambda':1, 'a':self._env._users[i]._service_a, 'b0':0, 'b':self._env._users[j]._service_b, 'c':self._env._users[j]._service_c, 'r':self._env._users[j]._service_r})
                        add = True
                        break
                    else:
                        k+=1
                if add:
                    break
    
    def select_two_max_constraint(self):
        self._env.re_initialize_num_server()
        self._constraints =[] #保存约束（一个客户端对的排队时延）的权重、其中包含的服务      
        max_delay, _2nd_max_delay = self._env.get_max_two_queuing_delay_in_interaction_delay()
        i = max_delay[1][0]
        j = max_delay[1][1]
        self._constraints.append(  {  'users':(i , j), 'lambda':1, 
            'a':self._env._users[i]._service_a, 'b0':0, 'b':self._env._users[j]._service_b, 'c':self._env._users[j]._service_c, 'r':self._env._users[j]._service_r})
        i = _2nd_max_delay[1][0]
        j = _2nd_max_delay[1][1]       
        self._constraints.append(  {  'users':(i , j), 'lambda':1, 
            'a':self._env._users[i]._service_a, 'b0':0, 'b':self._env._users[j]._service_b, 'c':self._env._users[j]._service_c, 'r':self._env._users[j]._service_r})        
  
  
    def surrogate(self):
        #print('--------surrogate-----------')
        delta = 0.00001
        self.select_two_max_constraint()
        k = 1
        f_lambda = - sys.maxsize
        while True:
            f_lambda_k = self.line_search() #lower bound
            #print('f_lambda_k{} - f_lambda{} = {} > initial upper bound * delta{}'.format(f_lambda_k ,f_lambda ,f_lambda_k -f_lambda, self._initial_upper_bound * delta))
            max_delay, user_i, user_j = self._env.get_max_queuing_delay_in_interaction_delay()
            if f_lambda_k - f_lambda > self._initial_upper_bound * delta:
                self._constraints.append(  {  'users':(user_i,user_j), 'lambda':1, 'a':self._env._users[user_i]._service_a, 'b0':0, 'b':self._env._users[user_j]._service_b, 'c':self._env._users[user_j]._service_c, 'r':self._env._users[user_j]._service_r})              
                f_lambda = f_lambda_k
                k += 1           
                '''
                lambda_sum = 0
                lambda_list = []
                for i in range(len(self._constraints)):
                    lambda_list.append(self._constraints[i]['lambda'])
                    lambda_sum += self._constraints[i]['lambda']         
                '''
                #print('--------add max delay : lambda_list={} lambda_sum={}'.format(lambda_list,lambda_sum))
                #print('constraints: ',self._constraints)                
                
                
            else:
                f_lambda = f_lambda_k
                '''
                lambda_sum = 0
                lambda_list = []
                for i in range(len(self._constraints)):
                    lambda_list.append(self._constraints[i]['lambda'])
                    lambda_sum += self._constraints[i]['lambda']   
                '''
                #print('-----------stop surrogate,  lambda_list={} lambda_sum={}'.format(lambda_list, lambda_sum))
                #print('constraints: ',self._constraints)
                return f_lambda
        return -1
        

    def set_num_server(self):
        self.initial_observation = self._env.observe()
        self.set_num_server_min_max_greedy()
        self._initial_upper_bound = self._max_delay
        self._start_time = time.time()
        self.surrogate()    
        self.get_running_time()
        #print('upper_bound =',self._upper_bound_list) 
        #print('lower_bound =',self._lower_bound_list) 
        
        self._upper_bound = min(self._upper_bound_list)
        self._lower_bound = max(self._lower_bound_list)      
        assert self.best_max_delay == self._upper_bound
        print('upper_bound =[', self._upper_bound, ']') 
        print('lower_bound =',self._lower_bound) 
        #self.plot_upper_and_lower_bound()
        
        #self.save_supervise_data()######################################################
        self.analyze_constraint_weight()
    
    def analyze_constraint_weight(self):
        self.get_constraint_weight()
        #print('self.constraint_weight : ',self.constraint_weight_list)
        num_none_zero = 0
        self.max_weight = -1
        self.min_weight = 2       

        for i in range(len(self.constraint_weight_list)):
            if self.constraint_weight_list[i] > 1e-5:
                num_none_zero += 1
                if self.constraint_weight_list[i] < self.min_weight:
                    self.min_weight = self.constraint_weight_list[i]
                if self.constraint_weight_list[i] > self.max_weight:
                    self.max_weight = self.constraint_weight_list[i]
        self.none_zero_ratio = 100 * num_none_zero/len(self.constraint_weight_list)
        
    
    def save_supervise_data(self):
        if self._initial_upper_bound >= self.best_max_delay:
            self.get_constraint_weight()
            data = {'observation' : self.initial_observation,
                    'constraint_weight' : self.constraint_weight_list }
            suffix = str(self._env._seed_sequence.entropy)[-8:]
            filename_name = './supervise_data/temporary/supervise_data_{}.json'.format(suffix)   
            with open(filename_name, 'w') as fid_json:            
                simplejson.dump(data, fid_json)               
    
    def get_topk_max_delay_index(self):
        delay_index_list = []
        for key, value in self.constraint_weight.items():
            user_i = key[0]
            user_j = key[1]
            delay_index = self._env.get_delay_index(user_i , user_j)
            i_a, i_b0, i_b, i_c, i_r  = self._env.get_service_index(user_i, user_j)
            delay_index_list.append({'delay_id': delay_index,
                              'service_id' : [float(i_a), float(i_b0), float(i_b), float(i_c), float(i_r)],
                              'weight':value})

        self.result_dict['top_k_delay_id'] = delay_index_list
 
    def get_result_dict(self):
        self.result_dict = {'max_delay':self._upper_bound, 'lower_bound':self._lower_bound, 
                            'running_time':self._running_time, 'none_zero_ratio':self.none_zero_ratio,
                            'max_weight':self.max_weight, 'min_weight':self.min_weight
                            }
        #self.get_topk_max_delay_index()
        return self.result_dict    
        
    def plot_upper_and_lower_bound(self):
        fontsize = 18
        linewidth = 2
        markersize = 8
        plt.rcParams.update({'font.size':fontsize, 'lines.linewidth':linewidth, 'lines.markersize':markersize, 'pdf.fonttype':42, 'ps.fonttype':42})
        fontsize_legend = 18
        color_list = ['#FF1F5B', '#009ADE',  '#F28522', '#58B272', '#AF58BA', '#A6761D','#1f77b4','#ff7f0e'] 
        marker_list = ['o', '^', 'X', 'd', 's', 'v', 'P',  '*','>','<','x']
        label_list = ['upper bound','lower bound','greedy']
        
        def plot(x, y):
            fig, ax = plt.subplots()
            max_y = -1
            min_y = 1000000
            for i in range(len(y)):
                plt.plot(x, y[i], label=label_list[i], marker=marker_list[i], color=color_list[i])    
                if max(y[i]) > max_y:
                    max_y = max(y[i])
                if min(y[i]) < min_y:
                    min_y = min(y[i])
            
            plt.legend(fontsize=fontsize_legend)    
            plt.xlabel('Num of solving')
            plt.ylabel('Max queuing delay')
            plt.grid(linestyle='--')
            plt.tight_layout()
            plt.ylim([min_y-1, max_y+1])
            plt.xticks(ticks=x)
            filename = 'Num of solving Max delay.pdf'
            #plt.savefig(filename, transparent=True, bbox_inches='tight', pad_inches=0.04)            
            plt.show()     
        
        x = np.arange(1,len(self._upper_bound_list)+1)
        y = []
        y.append(self._upper_bound_list)
        y.append(self._lower_bound_list)
        greedy = []
        for i in range(len(self._upper_bound_list)):
            greedy.append(self._initial_upper_bound)
        y.append(greedy)
        for i in range(len(y)):
            for j in range(len(self._upper_bound_list)):
                y[i][j]*=1000
        plot(x,y)        


class min_max_equal_weight(Algorithm):
    def __init__(self, env):
        Algorithm.__init__(self, env)

    def compute_weight(self):
        numerator = len(self._env._queuing_delay_without_duplicate) ** 2
        self._weight = {}

        for key, value in self._env._queuing_delay_without_duplicate.items():
            for service in value["service_set"]:
                _type, index = self._env.get_service_type_and_index(service)
                if (_type, index) in self._weight.keys():
                    self._weight[(_type, index)] += 1 / numerator
                else:
                    self._weight[(_type, index)] = 1 / numerator

    # def compute_weight(self):
    #     numerator = self._env._num_user * self._env._num_user
    #     self._weight = {}
    #     for i in range(self._env._num_user):
    #         for j in range(self._env._num_user):
    #             index_a = self._env._users[i]._service_a
    #             index_b = self._env._users[j]._service_b
    #             index_c = self._env._users[j]._service_c
    #             index_r = self._env._users[j]._service_r
    #             if ('a', index_a) in self._weight.keys():
    #                 self._weight[('a',index_a)] += 1/numerator
    #             else:
    #                 self._weight[('a',index_a)] = 1/numerator
    #
    #             if ('b0', 0) in self._weight.keys():
    #                 self._weight[('b0', 0)] += 1/numerator
    #             else:
    #                 self._weight[('b0', 0)] = 1/numerator
    #
    #             if ('b',index_b) in self._weight.keys():
    #                 self._weight[('b',index_b)] += 1/numerator
    #             else:
    #                 self._weight[('b',index_b)] = 1/numerator
    #
    #             if ('c', index_c) in self._weight.keys():
    #                 self._weight[('c', index_c)]+= 1/numerator
    #             else:
    #                 self._weight[('c', index_c)] = 1/numerator
    #
    #             if ('r', index_r) in self._weight.keys():
    #                 self._weight[('r', index_r)]+= 1/numerator
    #             else:
    #                 self._weight[('r', index_r)] = 1/numerator

    def get_max_utility_min_sum(self, services):
        services = sorted(services, key=lambda services : services['reduction']/services['price'],reverse=True)
        max_utility = None
        for k in range(len(services)):
            if services[k]['price'] >  self._env._cost_budget - self._cost:
                continue
            else:
                max_utility = services[k]
                break
        return max_utility       
    
    def set_num_server(self):
        self._start_time = time.time()   
        self.get_initial_max_queuing_delay()
        self.compute_weight()
        while self._cost < self._env._cost_budget:
            services = []
            for k in self._weight.keys():
                _type = k[0]
                index = k[1]
                if _type == 'a':
                    services.append({'type':'a', 'index':index, 
                                     'reduction': self._env._service_A[index].reduction_of_delay_when_add_a_server() * self._weight[('a', index)],
                                     'price':self._env._service_A[index]._price})                
                elif _type == 'b0':
                    services.append({'type':'b0', 'index':0, 
                                     'reduction': self._env._service_b0.reduction_of_delay_when_add_a_server() * self._weight[('b0', 0)],
                                     'price':self._env._service_b0._price})                    
                elif _type == 'b':
                    services.append({'type':'b', 'index':index, 
                                     'reduction': self._env._service_B[index].reduction_of_delay_when_add_a_server()  * self._weight[('b', index)],
                                     'price':self._env._service_B[index]._price})     
                elif _type == 'c':
                    services.append({'type':'c', 'index':index, 
                                     'reduction': self._env._service_C[index].reduction_of_delay_when_add_a_server() * self._weight[('c', index)],
                                     'price':self._env._service_C[index]._price  * len(self._env._service_C[index]._group)})         
                elif _type == 'r':
                    services.append({'type':'r', 'index':index, 
                                     'reduction': self._env._service_R[index].reduction_of_delay_when_add_a_server() * self._weight[('r', index)],
                                     'price':self._env._service_R[index]._price})
                
            max_utility = self.get_max_utility_min_sum(services)
            #print('==========add server for: ',max_utility)
            if max_utility:
                if max_utility['type'] == 'a':
                    self._env._service_A[max_utility['index']].update_num_server(self._env._service_A[max_utility['index']]._num_server + 1)
                elif max_utility['type'] == 'b0':
                    self._env._service_b0.update_num_server(self._env._service_b0._num_server + 1)
                elif max_utility['type'] == 'b':
                    self._env._service_B[max_utility['index']].update_num_server(self._env._service_B[max_utility['index']]._num_server + 1)
                elif max_utility['type'] == 'c':
                    self._env._service_C[max_utility['index']].update_num_server(self._env._service_C[max_utility['index']]._num_server + 1)
                elif max_utility['type'] == 'r':
                    self._env._service_R[max_utility['index']].update_num_server(self._env._service_R[max_utility['index']]._num_server + 1)
            else:
                break
            self._cost = self._env.compute_cost()   
        self.get_min_max_result()
        self.get_running_time()
        print(self._running_time)
        return   

class min_max_pulp(Algorithm):
    def __init__(self, env):
        Algorithm.__init__(self, env)     
    
    def compute_delay_reduction_when_add_n_server_per_service(self):
        self._delay_reduction_per_service = [] #二维矩阵,服务i在添加j个服务器后的时延减少量
        for i in range(self._env._num_service):
            _type, index  = self._env.get_service_type_and_index(i)
            delay_reduction = []

            max_num_server = int(self._env._budget_addition/self._env._price_list[i])
            for j in range(max_num_server+1):
                if _type == 'a':    
                    delay_reduction.append(self._env._service_A[index].reduction_of_delay_when_add_some_server(j))
                elif _type == 'b0':
                    delay_reduction.append(self._env._service_b0.reduction_of_delay_when_add_some_server(j))
                elif _type == 'b':
                    delay_reduction.append(self._env._service_B[index].reduction_of_delay_when_add_some_server(j))
                elif _type == 'c':
                    delay_reduction.append(self._env._service_C[index].reduction_of_delay_when_add_some_server(j))
                elif _type == 'r':           
                    delay_reduction.append(self._env._service_R[index].reduction_of_delay_when_add_some_server(j))
            self._delay_reduction_per_service.append(delay_reduction)              
    
    def set_num_server(self):
        self._start_time = time.time()   
        self.get_initial_max_queuing_delay()        
        
        self.compute_delay_reduction_when_add_n_server_per_service()
        model = pulp.LpProblem("Min_max_delay_model", pulp.LpMinimize) 
        
        x = [] #i服务增加j个服务器
        for i in range(self._env._num_service):
            _type, index  = self._env.get_service_type_and_index(i)
            #每个x有一个列表，元素是增加i个服务器的时延减少量，选择其中一个赋值为1，其他为0
            x_service = []
            max_num_server = int(self._env._budget_addition/self._env._price_list[i])
            for j in range(max_num_server+1):
                x_service.append(pulp.LpVariable("x_{}_{}_num_{}".format(_type, index, j), lowBound=0, upBound=1, cat = 'Binary'))
            x.append(x_service)
            model += sum([x[i][j] for j in range(len(x_service))]) == 1

        model += sum([   sum([ x[i][j] * j * self._env._price_list[i]  for j in range(int(self._env._budget_addition/self._env._price_list[i]) +1)])  for i in  range(self._env._num_service) ]) <= self._env._budget_addition

        # target
        M = pulp.LpVariable("M", lowBound=0, cat = 'Continuous')
        model +=  M

        #print(x)

        # target是M，即最大时延。那么要求所有优化后的时延都 <= M
        for user_i in range(self._env._num_user):
            for user_j in range(self._env._num_user):
                i_a, i_b0, i_b, i_c, i_r = self._env.get_service_index(user_i, user_j)
                model += self._env.compute_queuing_delay(user_i, user_j) -\
                    sum( [self._delay_reduction_per_service[i_a][j] * x[i_a][j] for j in range(len(self._delay_reduction_per_service[i_a]))]   ) -\
                    sum( [self._delay_reduction_per_service[i_b0][j] * x[i_b0][j] for j in range(len(self._delay_reduction_per_service[i_b0]))]   ) -\
                    sum( [self._delay_reduction_per_service[i_b][j] * x[i_b][j] for j in range(len(self._delay_reduction_per_service[i_b]))]   ) -\
                    sum( [self._delay_reduction_per_service[i_c][j] * x[i_c][j] for j in range(len(self._delay_reduction_per_service[i_c]))]   ) -\
                    sum( [self._delay_reduction_per_service[i_r][j] * x[i_r][j] for j in range(len(self._delay_reduction_per_service[i_r]))]   )  <= M

        #solver = pulp.CPLEX_PY(msg=False, warmStart=False)
        solver = pulp.PULP_CBC_CMD(msg=False, warmStart=False, timeLimit=1800) #timeLimit=600  : 10min
    
        model.solve(solver)  
        
        solution_list = []
        cost_pulp = 0
        for i in range(self._env._num_service):
            _type, index  = self._env.get_service_type_and_index(i)

            max_num_server = int(self._env._budget_addition/self._env._price_list[i])
            
            num = 0
            for j in range(max_num_server+1):     
                if x[i][j].value() ==1:
                    num = j
                    break
            cost_pulp += num * self._env._price_list[i]   

            while num * self._env._price_list[i]  >  self._env._cost_budget - self._cost:
                num -= 1
            assert num>=0
            
            if _type == 'a':
                self._env._service_A[index].update_num_server(self._env._service_A[index]._num_server + num)
            elif _type == 'b0':
                self._env._service_b0.update_num_server(self._env._service_b0._num_server + num)
            elif _type == 'b':
                self._env._service_B[index].update_num_server(self._env._service_B[index]._num_server + num)
            elif _type == 'c':
                self._env._service_C[index].update_num_server(self._env._service_C[index]._num_server + num)
            elif _type == 'r':
                self._env._service_R[index].update_num_server(self._env._service_R[index]._num_server + num)
            self._cost = self._env.compute_cost()
        self.get_min_max_result()
        self.get_running_time()          


class min_max_nn(Algorithm):
    
    def __init__(self, env):
        Algorithm.__init__(self, env)
        lr = 1e-4
        bs = 32
        n_step = 3        
        pattern = 'eval'
        self.agent = DQAgent([env], lr, bs, n_step, pattern)
    
    def set_num_server(self):
        self._start_time = time.time()
        self.agent.reset(0)
        self.get_initial_max_queuing_delay()
        while True:
            observation = self.agent.problem_list[self.agent.problem_id].observe()
            action = self.agent.act(observation)
            (reward, done) = self.agent.problem_list[self.agent.problem_id].act(action)
            if done:
                break        
        
        self.get_min_max_result()
        self.get_running_time()      

# class min_max_supervise(Algorithm):
#
#     def __init__(self, env):
#         Algorithm.__init__(self, env)
#         self.load_model()
#
#     def load_model(self):
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         self.model = my_S2V_QN_scheme3_supervise(self.device).to(self.device)
#         #para = torch.load('./supervise_checkpoint/model-969627/model-e=10000.pth')
#         #self.model.load_state_dict(para)
#
#     def get_constraint_weight(self):
#         obs = self._env.observe()
#         output  = self.model(obs).detach().cpu().numpy()
#         self.constraint_weight = output.flatten()
#
#     def get_service_weight(self):
#         self.service_weight = np.zeros(self._env._num_service)
#         constraint_index = 0
#         for i in range(self._env._num_user):
#             for j in range(self._env._num_user):
#                 i_a, i_b0, i_b, i_c, i_r = self._env.get_service_index(i, j)
#                 self.service_weight[i_a] += self.constraint_weight[constraint_index]
#                 self.service_weight[i_b0] += self.constraint_weight[constraint_index]
#                 self.service_weight[i_b] += self.constraint_weight[constraint_index]
#                 self.service_weight[i_c] += self.constraint_weight[constraint_index]
#                 self.service_weight[i_r] += self.constraint_weight[constraint_index]
#                 constraint_index += 1
#
#     def get_max_utility_min_sum(self, services):
#         services = sorted(services, key=lambda services : services['reduction']/services['price'],reverse=True)
#         max_utility = None
#         for k in range(len(services)):
#             if services[k]['price'] >  self._env._cost_budget - self._cost:
#                 continue
#             else:
#                 max_utility = services[k]
#                 break
#         return max_utility
#
#     def set_num_server(self):
#         self._start_time = time.time()
#         self.get_initial_max_queuing_delay()
#         self.get_constraint_weight()
#         self.get_service_weight()
#         while self._cost < self._env._cost_budget:
#             services = []
#             for i in range(self._env._num_service):
#                 _type, index = self._env.get_service_type_and_index(i)
#                 if _type == 'a':
#                     services.append({'type':'a', 'index':index,
#                                      'reduction': self._env._service_A[index].reduction_of_delay_when_add_a_server() * self.service_weight[i],
#                                      'price':self._env._service_A[index]._price})
#                 elif _type == 'b0':
#                     services.append({'type':'b0', 'index':0,
#                                      'reduction': self._env._service_b0.reduction_of_delay_when_add_a_server() * self.service_weight[i],
#                                      'price':self._env._service_b0._price})
#                 elif _type == 'b':
#                     services.append({'type':'b', 'index':index,
#                                      'reduction': self._env._service_B[index].reduction_of_delay_when_add_a_server()  * self.service_weight[i],
#                                      'price':self._env._service_B[index]._price})
#                 elif _type == 'c':
#                     price_c = 0
#                     if len(self._env._service_C[index]._group) > 0:
#                         price_c = self._env._service_C[index]._price  * len(self._env._service_C[index]._group)
#                     else:
#                         price_c = self._env._service_C[index]._price
#                     services.append({'type':'c', 'index':index,
#                                      'reduction': self._env._service_C[index].reduction_of_delay_when_add_a_server() * self.service_weight[i],
#                                      'price' : price_c})
#                 elif _type == 'r':
#                     services.append({'type':'r', 'index':index,
#                                      'reduction': self._env._service_R[index].reduction_of_delay_when_add_a_server() * self.service_weight[i],
#                                      'price':self._env._service_R[index]._price})
#
#             max_utility = self.get_max_utility_min_sum(services)
#             #print('==========add server for: ',max_utility)
#             if max_utility:
#                 if max_utility['type'] == 'a':
#                     self._env._service_A[max_utility['index']].update_num_server(self._env._service_A[max_utility['index']]._num_server + 1)
#                 elif max_utility['type'] == 'b0':
#                     self._env._service_b0.update_num_server(self._env._service_b0._num_server + 1)
#                 elif max_utility['type'] == 'b':
#                     self._env._service_B[max_utility['index']].update_num_server(self._env._service_B[max_utility['index']]._num_server + 1)
#                 elif max_utility['type'] == 'c':
#                     self._env._service_C[max_utility['index']].update_num_server(self._env._service_C[max_utility['index']]._num_server + 1)
#                 elif max_utility['type'] == 'r':
#                     self._env._service_R[max_utility['index']].update_num_server(self._env._service_R[max_utility['index']]._num_server + 1)
#             else:
#                 break
#             self._cost = self._env.compute_cost()
#         self.get_min_max_result()
#         self.get_running_time()
#         return
#
#
# #最小化总交互时延
class dynamic_programming(Algorithm):
    def __init__(self, env):
        Algorithm.__init__(self, env)
    
    def initialize_dp(self):
        self._num_service = self._env._num_service_a + 1 + self._env._num_service_b + self._env._num_service_c + self._env._num_service_r
        self.dp = np.zeros((self._num_service + 1, self._env._budget_addition + 1))
        self.dp_record = [[{'num_server': 0,'source':0} for j in range(self._env._budget_addition + 1)] for i in range(self._num_service + 1)]

    def set_num_server(self):
        self.get_delay_before()
        self.initialize_dp()
        for i in range(self._num_service):
            type = None
            index = -1
            weight_i = 0
            if i < self._env._num_service_a:
                type = 'a'
                index = i
                weight_i = self._env._service_A[index]._price
            elif i == self._env._num_service_a:
                type = 'b0'
                index = 0
                weight_i = self._env._service_b0._price
            elif i > self._env._num_service_a and i < self._env._num_service_a + 1 + self._env._num_service_b:
                type = 'b'
                index = i - self._env._num_service_a - 1
                weight_i = self._env._service_B[index]._price
            elif i >= self._env._num_service_a + 1 + self._env._num_service_b and i < self._env._num_service_a + 1 + self._env._num_service_b + self._env._num_service_c:
                type = 'c'
                index = i - self._env._num_service_a - 1 - self._env._num_service_b
                weight_i = self._env._service_C[index]._price * len(self._env._service_C[index]._group)
            else:
                type = 'r'
                index = i - self._env._num_service_a - 1 - self._env._num_service_b - self._env._num_service_c
                weight_i = self._env._service_R[index]._price


            #print('==============service {} id {} price{}'.format(type, index, weight_i))
            for j in range(self._env._budget_addition + 1):
                #print('==============budget{}'.format(j))
                #print('dp[i][j] = dp[i-1][j] = ',round(self.dp[i][j],2))
                self.dp[i+1][j] = self.dp[i][j]
                self.dp_record[i+1][j]['source'] = j
                k = 0
                while True:
                    if k * weight_i > j:
                        break
                    delay_reduction = 0.
                    if type == 'a':
                        delay_reduction = self._env._service_A[index].reduction_of_delay_when_add_some_server(k) * len(self._env._service_A[index]._users)*self._env._num_user
                    elif type == 'b0':
                        delay_reduction = self._env._service_b0.reduction_of_delay_when_add_some_server(k) * len(self._env._service_b0._users)*self._env._num_user
                    elif type == 'b':
                        delay_reduction = self._env._service_B[index].reduction_of_delay_when_add_some_server(k) * len(self._env._service_B[index]._users)*self._env._num_user
                    elif type == 'c':
                        delay_reduction = self._env._service_C[index].reduction_of_delay_when_add_some_server(k) * len(self._env._service_C[index]._users)*self._env._num_user
                    else:
                        delay_reduction = self._env._service_R[index].reduction_of_delay_when_add_some_server(k) * len(self._env._service_R[index]._users)*self._env._num_user
                    #self.dp[i+1][j] = max( self.dp[i+1][j], self.dp[i][j - k * weight_i] + delay_reduction) #???????????????????????重复
                    #print('服务器数量k：{}  dp[i-1][j - k * weight[i]] + value[i]={}'.format(k,round(self.dp[i][j - k * weight_i] + delay_reduction,2)))
                    if self.dp[i+1][j] <= self.dp[i][j - k * weight_i] + delay_reduction:
                        self.dp[i+1][j] = self.dp[i][j - k * weight_i] + delay_reduction
                        self.dp_record[i+1][j]['num_server'] = k
                        self.dp_record[i+1][j]['source'] = j - k * weight_i
                    k += 1
                #print('dp[i][j] = ',round(self.dp[i+1][j],2))
        self.get_dp_result()
        
        self.get_result_after()
    
    def get_dp_result(self):
        self._delay_reduction = self.dp[self._num_service][self._env._budget_addition]
        i = self._num_service
        j = self._env._budget_addition
        while i > 0:
            if i - 1 < self._env._num_service_a:
                type = 'a'
                index = i - 1
                self._env._service_A[index].update_num_server(self._env._service_A[index]._num_server + self.dp_record[i][j]['num_server'])
            elif i - 1 == self._env._num_service_a:
                type = 'b0'
                index = 0
                self._env._service_b0.update_num_server(self._env._service_b0._num_server + self.dp_record[i][j]['num_server'])
            elif i - 1 > self._env._num_service_a and i - 1 < self._env._num_service_a + 1 + self._env._num_service_b:
                type = 'b'
                index = i - self._env._num_service_a - 1 - 1
                self._env._service_B[index].update_num_server(self._env._service_B[index]._num_server + self.dp_record[i][j]['num_server'])
            elif i - 1 >= self._env._num_service_a + 1 + self._env._num_service_b and i - 1 < self._env._num_service_a + 1 + self._env._num_service_b + self._env._num_service_c:
                type = 'c'
                index = i - self._env._num_service_a - 1 - self._env._num_service_b - 1
                self._env._service_C[index].update_num_server(self._env._service_C[index]._num_server + self.dp_record[i][j]['num_server'])
            else:
                type = 'r'
                index = i - self._env._num_service_a - 1 - self._env._num_service_b - self._env._num_service_c - 1
                self._env._service_R[index].update_num_server(self._env._service_R[index]._num_server + self.dp_record[i][j]['num_server'])
            
            j = self.dp_record[i][j]['source']
            i -= 1

class greedy(Algorithm):
    def __init__(self, env):
        Algorithm.__init__(self, env)

    def get_max_utility(self, services):
        services = sorted(services, key=lambda services : services['reduction']/services['price'],reverse=True)
        max_utility = None
        for k in range(len(services)):
            if services[k]['price'] >  self._env._cost_budget - self._cost:
                continue
            else:
                max_utility = services[k]
                break
        if max_utility:
            self._delay_reduction += max_utility['reduction']
        return max_utility

    def set_num_server(self):
        self.get_delay_before()
        self._num_service = self._env._num_service_a + 1 + self._env._num_service_b + self._env._num_service_c + self._env._num_service_r
        while self._cost < self._env._cost_budget:
            services = []
            for i in range(self._num_service):
                type = None
                index = -1
                if i < self._env._num_service_a:
                    type = 'a'
                    index = i
                    services.append({'type':type, 'index':index, 
                    'reduction': self._env._service_A[index].reduction_of_delay_when_add_a_server()* len(self._env._service_A[index]._users)*self._env._num_user,
                    'price':self._env._service_A[index]._price})
                elif i == self._env._num_service_a:
                    type = 'b0'
                    index = 0
                    services.append({'type':type, 'index':index, 
                    'reduction': self._env._service_b0.reduction_of_delay_when_add_a_server()* len(self._env._service_b0._users)*self._env._num_user, 
                    'price':self._env._service_b0._price})
                elif i > self._env._num_service_a and i < self._env._num_service_a + 1 + self._env._num_service_b:
                    type = 'b'
                    index = i - self._env._num_service_a - 1
                    services.append({'type':type, 'index':index, 
                    'reduction': self._env._service_B[index].reduction_of_delay_when_add_a_server()* len(self._env._service_B[index]._users)*self._env._num_user, 
                    'price':self._env._service_B[index]._price})
                elif i >= self._env._num_service_a + 1 + self._env._num_service_b and i < self._env._num_service_a + 1 + self._env._num_service_b + self._env._num_service_c:
                    type = 'c'
                    index = i - self._env._num_service_a - 1 - self._env._num_service_b
                    services.append({'type':type, 'index':index, 
                    'reduction': self._env._service_C[index].reduction_of_delay_when_add_a_server()* len(self._env._service_C[index]._users)*self._env._num_user, 
                    'price':self._env._service_C[index]._price * len(self._env._service_C[index]._group)})
                else:
                    type = 'r'
                    index = i - self._env._num_service_a - 1 - self._env._num_service_b - self._env._num_service_c
                    services.append({'type':type, 'index':index, 
                    'reduction': self._env._service_R[index].reduction_of_delay_when_add_a_server()* len(self._env._service_R[index]._users)*self._env._num_user,
                    'price':self._env._service_R[index]._price})
            max_utility = self.get_max_utility(services)
            if max_utility['type'] == 'a':
                self._env._service_A[max_utility['index']].update_num_server(self._env._service_A[max_utility['index']]._num_server + 1)
            elif max_utility['type'] == 'b0':
                self._env._service_b0.update_num_server(self._env._service_b0._num_server + 1)
            elif max_utility['type'] == 'b':
                self._env._service_B[max_utility['index']].update_num_server(self._env._service_B[max_utility['index']]._num_server + 1)
            elif max_utility['type'] == 'c':
                self._env._service_C[max_utility['index']].update_num_server(self._env._service_C[max_utility['index']]._num_server + 1)
            elif max_utility['type'] == 'r':
                self._env._service_R[max_utility['index']].update_num_server(self._env._service_R[max_utility['index']]._num_server + 1)
            else:
                break
            self._cost = self._env.compute_cost()
        self.get_result_after()
        
        
class random(Algorithm):
    def __init__(self, env):
        Algorithm.__init__(self, env)

    def get_random_service(self):
        for k in range(len(services)):
            if services[k]['price'] >  self._env._cost_budget - self._cost:
                continue
            else:
                max_utility = services[k]
                break
        if max_utility:
            self._delay_reduction += max_utility['reduction']
        return max_utility

    def set_num_server(self):
        self.get_delay_before()
        self._num_service = self._env._num_service_a + 1 + self._env._num_service_b + self._env._num_service_c + self._env._num_service_r
        while self._cost < self._env._cost_budget:
            random_service={'index':-1,'type':None,'reduction':0,'price':0}
            while True:
                i = self._env._rng.choice(self._num_service)
                type = None
                index = -1
                reduction = 0
                price = 0
                if i < self._env._num_service_a:
                    type = 'a'
                    index = i
                    reduction = self._env._service_A[index].reduction_of_delay_when_add_a_server()* len(self._env._service_A[index]._users)*self._env._num_user
                    price = self._env._service_A[index]._price
                elif i == self._env._num_service_a:
                    type = 'b0'
                    index = 0  
                    reduction = self._env._service_b0.reduction_of_delay_when_add_a_server()* len(self._env._service_b0._users)*self._env._num_user
                    price = self._env._service_b0._price
                elif i > self._env._num_service_a and i < self._env._num_service_a + 1 + self._env._num_service_b:
                    type = 'b'
                    index = i - self._env._num_service_a - 1
                    reduction = self._env._service_B[index].reduction_of_delay_when_add_a_server()* len(self._env._service_B[index]._users)*self._env._num_user
                    price = self._env._service_B[index]._price
                elif i >= self._env._num_service_a + 1 + self._env._num_service_b and i < self._env._num_service_a + 1 + self._env._num_service_b + self._env._num_service_c:
                    type = 'c'
                    index = i - self._env._num_service_a - 1 - self._env._num_service_b
                    reduction = self._env._service_C[index].reduction_of_delay_when_add_a_server()* len(self._env._service_C[index]._users)*self._env._num_user
                    price = self._env._service_C[index]._price * len(self._env._service_C[index]._group)
                else:
                    type = 'r'
                    index = i - self._env._num_service_a - 1 - self._env._num_service_b - self._env._num_service_c
                    reduction = self._env._service_R[index].reduction_of_delay_when_add_a_server()* len(self._env._service_R[index]._users)*self._env._num_user
                    price = self._env._service_R[index]._price
                if price >  self._env._cost_budget - self._cost:
                    continue
                else:
                    random_service['index']=index
                    random_service['type']=type
                    random_service['reduction']=reduction
                    random_service['price']=price
                    self._delay_reduction += random_service['reduction']    
                    break

            if random_service['type'] == 'a':
                self._env._service_A[random_service['index']].update_num_server(self._env._service_A[random_service['index']]._num_server + 1)
            elif random_service['type'] == 'b0':
                self._env._service_b0.update_num_server(self._env._service_b0._num_server + 1)
            elif random_service['type'] == 'b':
                self._env._service_B[random_service['index']].update_num_server(self._env._service_B[random_service['index']]._num_server + 1)
            elif random_service['type'] == 'c':
                self._env._service_C[random_service['index']].update_num_server(self._env._service_C[random_service['index']]._num_server + 1)
            elif random_service['type'] == 'r':
                self._env._service_R[random_service['index']].update_num_server(self._env._service_R[random_service['index']]._num_server + 1)
            else:
                break
            self._cost = self._env.compute_cost()
        self.get_result_after()