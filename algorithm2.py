import numpy as np
import sys
from time import time
import pulp

from env2 import *

import simplejson


class Algorithm:
    def __init__(self, env):
        self._env = env
        self._max_delay = 0

        self._start_time = None
        self._end_time = None

    def get_running_time(self):
        self._end_time = time()
        self._running_time = (self._end_time - self._start_time) * 1000  # ms

    def set_num_server(self):
        pass

    # min max
    def get_initial_max_queuing_delay(self):
        self._cost = self._env.compute_cost()
        max_delay, user_i, user_j = self._env.get_max_queuing_delay_in_interaction_delay()
        self._max_queuing_delay_initial = max_delay
        print('initial max queuing delay: [', self._max_queuing_delay_initial, '] cost ', self._cost, ' budget ',
              self._env._cost_budget)

    # min max
    def get_min_max_result(self):  # min max
        cost = self._env.compute_cost()
        max_delay, user_i, user_j = self._env.get_max_queuing_delay_in_interaction_delay()
        self._max_delay = max_delay
        print('final max queuing delay [', self._max_delay, '] cost ', self._cost, ' budget ',
              self._env._cost_budget)
        assert cost <= self._env._cost_budget

    def get_result_dict(self):
        self.result_dict = {'max_delay': self._max_delay, 'running_time': self._running_time}
        return self.result_dict


IF_DEBUG = False


class min_max_greedy(Algorithm):
    def __init__(self, env):
        Algorithm.__init__(self, env)

    def set_num_server(self):
        self._start_time = time()
        self.get_initial_max_queuing_delay()

        while self._cost < self._env._cost_budget:
            max_delay, user_i, user_j = self._env.get_max_queuing_delay_in_interaction_delay()

            user_i = self._env._users[user_i]
            user_j = self._env._users[user_j]
            service_type = ["a", "b0", "b", "c", "r"]
            service_list = [
                self._env._service_A[user_i._service_a],
                self._env._service_b0,
                self._env._service_B[user_j._service_b],
                self._env._service_C[user_j._service_c][user_j._sub_service_c],
                self._env._service_R[user_j._service_r][user_j._sub_service_r]
            ]

            services = []
            for i, s_type in enumerate(service_type):
                services.append({"service": s_type,
                                 "reduction": service_list[i].reduction_of_delay_when_add_a_server(),
                                 "price": service_list[i]._price})

            if IF_DEBUG:
                a, b0, b, c, sub_c, r, sub_r = self._env.get_service_index(user_i._id, user_j._id)
                indices = (a, b0, b, (c, sub_c), (r, sub_r))
                print("service indices: {}".format(indices))
                print("max_delay: {}, users: ({}, {})".format(max_delay, user_i._id, user_j._id))
                print("services: {}".format(services))

            max_utility = self.get_max_utility(services)

            if IF_DEBUG:
                print("max_utility: {}".format(max_utility))

            if max_utility is not None:
                selected_service = service_list[service_type.index(max_utility)]
                selected_service.update_num_server(selected_service._num_server + 1)
            else:
                break
            self._cost = self._env.compute_cost()

            if IF_DEBUG:
                print("cost: {}, cost_budget: {}\n".format(self._cost, self._env._cost_budget))

        self.get_min_max_result()
        self.get_running_time()

    def get_max_utility(self, services):
        services = sorted(services, key=lambda services: services["reduction"] / services["price"], reverse=True)
        max_utility = None

        for k in range(len(services)):
            if services[k]["price"] > self._env._cost_budget - self._cost:
                continue
            else:
                max_utility = services[k]["service"]
                break
        return max_utility


class min_max_pulp(Algorithm):
    def __init__(self, env):
        Algorithm.__init__(self, env)
        self.services_to_dict()

    # 将服务记录成字典形式
    # key = (type, id), value = service ---> 只是引用（浅拷贝）
    def services_to_dict(self):
        self._services_dict = {}

        for service in self._env._service_A:
            self._services_dict[("a", service._id)] = service

        self._services_dict[("b0", 0)] = self._env._service_b0

        for service in self._env._service_B:
            self._services_dict[("b", service._id)] = service

        for i in range(len(self._env._service_C)):
            for service in self._env._service_C[i]:
                self._services_dict[("c", (service._id, service._sub_id))] = service

        for i in range(len(self._env._service_R)):
            for service in self._env._service_R[i]:
                self._services_dict[("r", (service._id, service._sub_id))] = service

    def compute_delay_reduction_when_add_n_server_per_service(self):
        self._delay_reductions_each_service = {}
        for key, service in self._services_dict.items():
            delay_reduction = []
            max_num_server = int(self._env._budget_addition / service._price)
            for num_server in range(max_num_server + 1):
                delay_reduction.append(service.reduction_of_delay_when_add_some_server(num_server))
            self._delay_reductions_each_service[key] = delay_reduction

    def set_num_server(self):
        self._start_time = time()
        self.get_initial_max_queuing_delay()

        self.compute_delay_reduction_when_add_n_server_per_service()
        model = pulp.LpProblem("Min_max_delay_model", pulp.LpMinimize)

        # i服务增加j个服务器
        x = {}
        for key, service in self._services_dict.items():
            # 每个x有一个列表，元素是增加i个服务器的时延减少量，选择其中一个赋值为1，其他为0
            x_service = []
            max_num_server = int(self._env._budget_addition / service._price)
            for num_server in range(max_num_server + 1):
                if service._type == "c" or service._type == "r":
                    x_service.append(pulp.LpVariable(
                        "x_{}_{}_{}_num_{}".format(service._type, service._id, service._sub_id, num_server),
                        lowBound=0, upBound=1, cat='Binary'))
                else:
                    x_service.append(pulp.LpVariable(
                        "x_{}_{}_num_{}".format(service._type, service._id, num_server),
                        lowBound=0, upBound=1, cat='Binary'))
            x[key] = x_service

            # 每个服务的服务器个数是一个确定的值
            model += sum([x[key][j] for j in range(len(x_service))]) == 1

        # 增加服务器的开销之和要小于总预算
        model += sum([
            sum([
                x[key][num_server] * num_server * service_._price
                for num_server in range(int(self._env._budget_addition / service_._price) + 1)])
            for key, service_ in self._services_dict.items()
        ]) <= self._env._budget_addition

        # target: 最小化M值，而所有优化后的时延都要小于M
        M = pulp.LpVariable("M", lowBound=0, cat="Continuous")
        model += M

        for user_i in range(self._env._num_user):
            for user_j in range(self._env._num_user):
                a, b0, b, c, sub_c, r, sub_r = self._env.get_service_index(user_i, user_j)
                model += self._env.compute_queuing_delay(user_i, user_j) - \
                         sum([self._delay_reductions_each_service[("a", a)][j] * x[("a", a)][j] for j in
                              range(len(self._delay_reductions_each_service[("a", a)]))]) - \
                         sum([self._delay_reductions_each_service[("b0", b0)][j] * x[("b0", b0)][j] for j in
                              range(len(self._delay_reductions_each_service[("b0", b0)]))]) - \
                         sum([self._delay_reductions_each_service[("b", b)][j] * x[("b", b)][j] for j in
                              range(len(self._delay_reductions_each_service[("b", b)]))]) - \
                         sum([self._delay_reductions_each_service[("c", (c, sub_c))][j] * x[("c", (c, sub_c))][j] for j
                              in range(len(self._delay_reductions_each_service[("c", (c, sub_c))]))]) - \
                         sum([self._delay_reductions_each_service[("r", (r, sub_r))][j] * x[("r", (r, sub_r))][j] for j
                              in range(len(self._delay_reductions_each_service[("r", (r, sub_r))]))]) <= M

        solver = pulp.PULP_CBC_CMD(msg=False, warmStart=False, timeLimit=1800)  # timeLimit=600 ===> 10min

        model.solve(solver)

        cost_pulp = 0
        for key, service_ in self._services_dict.items():
            max_num_server = int(self._env._budget_addition / service_._price)

            num_server = 0
            for i in range(max_num_server + 1):
                if x[key][i].value() == 1:
                    num_server = i
                    break
            cost_pulp += num_server * service_._price

            # ?
            while num_server * service_._price > self._env._cost_budget - self._cost:
                num_server -= 1
            assert num_server >= 0

            service_.update_num_server(service_._num_server + num_server)

            self._cost = self._env.compute_cost()

        self.get_min_max_result()
        self.get_running_time()

        """ 
            Debug
        """
        print("\n------------- Info -------------")
        print("\n------------ services -----------------")
        for key, service_ in self._services_dict.items():
            print("{}: {}".format(key, service_._id))
        print("\n------------ reductions ----------------")
        for key, reductions in self._delay_reductions_each_service.items():
            print("{}: {}".format(key, reductions))
        print("\n------------ x -------------------")
        for key, value in x.items():
            print("{}: {}".format(key, value))
            for x_ in value:
                if x_.value() == 1:
                    print("num_server: {}".format(x_))


class min_max_equal_weight(Algorithm):
    def __init__(self, env):
        Algorithm.__init__(self, env)
        self._service_list = {
            "a": self._env._service_A,
            "b0": self._env._service_b0,
            "b": self._env._service_B,
            "c": self._env._service_C,
            "r": self._env._service_R
        }

    def set_num_server(self):
        self._start_time = time()
        self.get_initial_max_queuing_delay()
        self.compute_weight()
        while self._cost < self._env._cost_budget:
            services = []
            for key in self._weight.keys():  # key = (type, id)
                service = None
                if key[0] == "a" or key[0] == "b":
                    service = self._service_list[key[0]][key[1]]
                elif key[0] == "b0":
                    service = self._service_list[key[0]]
                else:
                    service = self._service_list[key[0]][key[1][0]][key[1][1]]

                services.append({
                    "type": key[0],
                    "id": key[1],
                    "reduction": service.reduction_of_delay_when_add_a_server() * self._weight[key],
                    "price": service._price
                })

            max_utility = self.get_max_utility_min_sum(services)

            if IF_DEBUG:
                print(max_utility)

            if max_utility is not None:
                s_type = max_utility["type"]
                s_id = max_utility["id"]
                if s_type == "a" or s_type == "b":
                    self._service_list[s_type][s_id].update_num_server(self._service_list[s_type][s_id]._num_server + 1)
                elif s_type == "b0":
                    self._service_list["b0"].update_num_server(self._service_list["b0"]._num_server + 1)
                else:
                    self._service_list[s_type][s_id[0]][s_id[1]].update_num_server(
                        self._service_list[s_type][s_id[0]][s_id[1]]._num_server + 1)
            else:
                break
            self._cost = self._env.compute_cost()

        self.get_min_max_result()
        self.get_running_time()

    def compute_weight(self):
        numerator = len(self._env._queuing_delay_without_duplicate) ** 2
        self._weight = {}

        service_types = ["a", "b0", "b", "c", "r"]
        for key, value in self._env._queuing_delay_without_duplicate.items():
            for i, service_id in enumerate(value["service_set"]):
                if (service_types[i], service_id) in self._weight.keys():
                    self._weight[(service_types[i], service_id)] += 1 / numerator
                else:
                    self._weight[(service_types[i], service_id)] = 1 / numerator

        if IF_DEBUG:
            for key, value in self._weight.items():
                print("{}: {}".format(key, value))

    def get_max_utility_min_sum(self, services):
        services = sorted(services, key=lambda services: services["reduction"] / services["price"], reverse=True)
        max_utility = None
        for k in range(len(services)):
            if services[k]["price"] > self._env._cost_budget - self._cost:
                continue
            else:
                max_utility = services[k]
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

    def set_num_server(self):
        self.set_num_server_min_max_greedy()
        self._initial_upper_bound = self._max_delay
        self._start_time = time()
        self.surrogate()
        self.get_running_time()

        self._upper_bound = min(self._upper_bound_list)
        self._lower_bound = max(self._lower_bound_list)
        assert self.best_max_delay == self._upper_bound
        print('upper_bound =[', self._upper_bound, ']')
        print('lower_bound =', self._lower_bound)

        # self.analyze_constraint_weight()

    def set_num_server_min_max_greedy(self):
        self.get_initial_max_queuing_delay()
        while self._cost < self._env._cost_budget:
            max_delay, user_i, user_j = self._env.get_max_queuing_delay_in_interaction_delay()
            user_i = self._env._users[user_i]
            user_j = self._env._users[user_j]
            service_type = ["a", "b0", "b", "c", "r"]
            service_list = [
                self._env._service_A[user_i._service_a],
                self._env._service_b0,
                self._env._service_B[user_j._service_b],
                self._env._service_C[user_j._service_c][user_j._sub_service_c],
                self._env._service_R[user_j._service_r][user_j._sub_service_r]
            ]

            services = []
            for i, s_type in enumerate(service_type):
                services.append({"service": s_type,
                                 "reduction": service_list[i].reduction_of_delay_when_add_a_server(),
                                 "price": service_list[i]._price})

            if IF_DEBUG:
                a, b0, b, c, sub_c, r, sub_r = self._env.get_service_index(user_i._id, user_j._id)
                indices = (a, b0, b, (c, sub_c), (r, sub_r))
                print("service indices: {}".format(indices))
                print("max_delay: {}, users: ({}, {})".format(max_delay, user_i._id, user_j._id))
                print("services: {}".format(services))

            max_utility = self.get_max_utility(services)

            if IF_DEBUG:
                print("max_utility: {}".format(max_utility))

            if max_utility is not None:
                selected_service = service_list[service_type.index(max_utility)]
                selected_service.update_num_server(selected_service._num_server + 1)
            else:
                break
            self._cost = self._env.compute_cost()

            if IF_DEBUG:
                print("cost: {}, cost_budget: {}\n".format(self._cost, self._env._cost_budget))

        self.get_min_max_result()

    # min max
    def get_initial_max_queuing_delay(self):
        self._cost = self._env.compute_cost()
        max_delay, user_i, user_j = self._env.get_max_queuing_delay_in_interaction_delay()
        self._max_queuing_delay_initial = max_delay
        print('initial max queuing delay: [', self._max_queuing_delay_initial, '] cost ', self._cost, ' budget ',
              self._env._cost_budget)

    def get_max_utility(self, services):
        services = sorted(services, key=lambda services: services["reduction"] / services["price"], reverse=True)
        max_utility = None

        for k in range(len(services)):
            if services[k]["price"] > self._env._cost_budget - self._cost:
                continue
            else:
                max_utility = services[k]["service"]
                break
        return max_utility

    def surrogate(self):
        delta = 0.00001
        self.select_two_max_constraint()
        k = 1
        f_lambda = - sys.maxsize
        while True:
            f_lambda_k = self.line_search()  # lower bound
            max_delay, user_i, user_j = self._env.get_max_queuing_delay_in_interaction_delay()
            if f_lambda_k - f_lambda > self._initial_upper_bound * delta:
                self._constraints.append(
                    {
                        'users': (user_i, user_j),
                        'lambda': 1,
                        'a': self._env._users[user_i]._service_a,
                        'b0': 0,
                        'b': self._env._users[user_j]._service_b,
                        'c': (self._env._users[user_j]._service_c, self._env._users[user_j]._sub_service_c),
                        'r': (self._env._users[user_j]._service_r, self._env._users[user_j]._sub_service_r)
                    }
                )
                f_lambda = f_lambda_k
                k += 1
            else:
                f_lambda = f_lambda_k
                return f_lambda

        return -1  # 多余

    def select_two_max_constraint(self):
        self._env.re_initialize_num_server()

        # 保存约束（一个客户端对的排队时延）的权重、其中包含的服务
        self._constraints = []
        max_delay, _2nd_max_delay = self._env.get_max_two_queuing_delay_in_interaction_delay()

        i = max_delay[1][0]
        j = max_delay[1][1]
        self._constraints.append(
            {
                "users": (i, j),
                "lambda": 1,
                "a": self._env._users[i]._service_a,
                "b0": 0,
                "b": self._env._users[j]._service_b,
                "c": (self._env._users[j]._service_c, self._env._users[j]._sub_service_c),
                "r": (self._env._users[j]._service_r, self._env._users[j]._sub_service_r)
            }
        )

        i = _2nd_max_delay[1][0]
        j = _2nd_max_delay[1][1]
        self._constraints.append(
            {
                'users': (i, j),
                'lambda': 1,
                'a': self._env._users[i]._service_a,
                'b0': 0,
                'b': self._env._users[j]._service_b,
                "c": (self._env._users[j]._service_c, self._env._users[j]._sub_service_c),
                "r": (self._env._users[j]._service_r, self._env._users[j]._sub_service_r)
            }
        )

    def line_search(self):
        epsilon = 0.05
        delta = 0.05
        lambda_low = 0.
        lambda_high = 1.
        k = 1
        f_lambda = 0
        lambda_k = 0
        while True:
            if lambda_high - lambda_low < epsilon:
                break
            lambda_k = lambda_low + (lambda_high - lambda_low) / 2

            f_lambda_delta = self.compute_min_sum_allocation_solution(lambda_k + delta)
            f_lambda = self.compute_min_sum_allocation_solution(lambda_k)

            self._lower_bound_list.append(f_lambda)
            self._upper_bound_list.append(self._max_delay)

            if self.best_max_delay > self._max_delay:
                self.best_max_delay = self._max_delay
                # self.save_constraint_weight(lambda_k)

            if (f_lambda_delta - f_lambda) / delta <= 0:
                lambda_high = lambda_k
            else:
                lambda_low = lambda_k
            k += 1

        w = 0
        for i in range(len(self._constraints)):
            if w == len(self._constraints) - 1:
                self._constraints[i]['lambda'] *= (1 - lambda_k)
            else:
                self._constraints[i]['lambda'] *= lambda_k
            w += 1
        return f_lambda

    def compute_min_sum_allocation_solution(self, lambda_k):
        self._env.re_initialize_num_server()
        self._cost = self._env.compute_cost()

        while self._cost < self._env._cost_budget:
            services = {}
            w = 0
            for constraint in self._constraints:
                weight = lambda_k
                if w == len(self._constraints) - 1:
                    weight = 1 - lambda_k

                if ("a", constraint["a"]) in services.keys():
                    services[('a', constraint['a'])]['reduction'] += self._env._service_A[constraint[
                        'a']].reduction_of_delay_when_add_a_server() * constraint['lambda'] * weight
                else:
                    services[('a', constraint['a'])] = {
                        'type': 'a',
                        'index': constraint['a'],
                        'reduction': self._env._service_A[constraint['a']].reduction_of_delay_when_add_a_server() *
                                     constraint['lambda'] * weight,
                        'price': self._env._service_A[constraint['a']]._price
                    }

                if ('b0', 0) in services.keys():
                    services[('b0', 0)]['reduction'] += self._env._service_b0.reduction_of_delay_when_add_a_server() * \
                                                        constraint['lambda'] * weight
                else:
                    services[('b0', 0)] = {
                        'type': 'b0',
                        'index': 0,
                        'reduction': self._env._service_b0.reduction_of_delay_when_add_a_server() * constraint[
                            'lambda'] * weight,
                        'price': self._env._service_b0._price
                    }

                if ('b', constraint['b']) in services.keys():
                    services[('b', constraint['b'])]['reduction'] += self._env._service_B[constraint[
                        'b']].reduction_of_delay_when_add_a_server() * constraint['lambda'] * weight
                else:
                    services[('b', constraint['b'])] = {
                        'type': 'b',
                        'index': constraint['b'],
                        'reduction': self._env._service_B[constraint['b']].reduction_of_delay_when_add_a_server() *
                                     constraint['lambda'] * weight,
                        'price': self._env._service_B[constraint['b']]._price
                    }

                # constraint['c] = (c, sub_c)
                if ('c', constraint['c']) in services.keys():
                    services[('c', constraint['c'])]['reduction'] += self._env._service_C[constraint['c'][0]][
                                                                         constraint['c'][
                                                                             1]].reduction_of_delay_when_add_a_server() * \
                                                                     constraint['lambda'] * weight
                else:
                    services[('c', constraint['c'])] = {
                        'type': 'c',
                        'index': constraint['c'],
                        'reduction': self._env._service_C[constraint['c'][0]][
                                         constraint['c'][1]].reduction_of_delay_when_add_a_server() *
                                     constraint['lambda'] * weight,
                        'price': self._env._service_C[constraint['c'][0]][constraint['c'][1]]._price
                    }

                if ('r', constraint['r']) in services.keys():
                    services[('r', constraint['r'])]['reduction'] += self._env._service_R[constraint['r'][0]][
                                                                         constraint['r'][
                                                                             1]].reduction_of_delay_when_add_a_server() * \
                                                                     constraint['lambda'] * weight
                else:
                    services[('r', constraint['r'])] = {
                        'type': 'r',
                        'index': constraint['r'],
                        'reduction': self._env._service_R[constraint['r'][0]][
                                         constraint['r'][1]].reduction_of_delay_when_add_a_server() *
                                     constraint['lambda'] * weight,
                        'price': self._env._service_R[constraint['r'][0]][constraint['r'][1]]._price
                    }

                w += 1

            max_utility = self.get_max_utility_min_sum(services)
            if max_utility:
                if max_utility['type'] == 'a':
                    self._env._service_A[max_utility['index']].update_num_server(
                        self._env._service_A[max_utility['index']]._num_server + 1)
                elif max_utility['type'] == 'b0':
                    self._env._service_b0.update_num_server(self._env._service_b0._num_server + 1)
                elif max_utility['type'] == 'b':
                    self._env._service_B[max_utility['index']].update_num_server(
                        self._env._service_B[max_utility['index']]._num_server + 1)
                elif max_utility['type'] == 'c':
                    self._env._service_C[max_utility['index'][0]][max_utility['index'][1]].update_num_server(
                        self._env._service_C[max_utility['index'][0]][max_utility['index'][1]]._num_server + 1)
                elif max_utility['type'] == 'r':
                    self._env._service_R[max_utility['index'][0]][max_utility['index'][1]].update_num_server(
                        self._env._service_R[max_utility['index'][0]][max_utility['index'][1]]._num_server + 1)
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
                weight = 1 - lambda_k
            f_lambda += self._env.compute_queuing_delay(constraint['users'][0], constraint['users'][1]) * constraint[
                'lambda'] * weight
            w += 1

        return f_lambda

    def get_max_utility_min_sum(self, services):
        value_list = []
        for v in services.values():
            value_list.append(v)
        value_list = sorted(value_list, key=lambda value_list: value_list['reduction'] / value_list['price'],
                            reverse=True)

        max_utility = None
        for k in range(len(value_list)):
            if value_list[k]['price'] > self._env._cost_budget - self._cost:
                continue
            else:
                max_utility = value_list[k]
                break
        return max_utility


# equal weight dp
class min_max_equal_weight_dp(Algorithm):
    def __init__(self, env):
        Algorithm.__init__(self, env)
        self.service_to_list()

    def service_to_list(self):
        self._services_list = []
        for service in self._env._service_A:
            self._services_list.append(service)

        self._services_list.append(self._env._service_b0)

        for service in self._env._service_B:
            self._services_list.append(service)

        for i in range(self._env._num_service_c):
            for service in self._env._service_C[i]:
                self._services_list.append(service)

        for i in range(self._env._num_service_r):
            for service in self._env._service_R[i]:
                self._services_list.append(service)

    def compute_weight(self):
        numerator = len(self._env._queuing_delay_without_duplicate) ** 2
        self._weight = {}

        service_types = ["a", "b0", "b", "c", "r"]
        for key, value in self._env._queuing_delay_without_duplicate.items():
            for i, service_id in enumerate(value["service_set"]):
                if (service_types[i], service_id) in self._weight.keys():
                    self._weight[(service_types[i], service_id)] += 1 / numerator
                else:
                    self._weight[(service_types[i], service_id)] = 1 / numerator

    def initialize_dp(self):
        self._num_service = len(self._services_list)
        self.dp = np.zeros((self._num_service + 1, self._env._budget_addition + 1))
        self.dp_record = [[{'num_server': 0,'source':0} for j in range(self._env._budget_addition + 1)] for i in range(self._num_service + 1)]

    def set_num_server(self):
        self._start_time = time()
        self.get_initial_max_queuing_delay()
        self.compute_weight()
        self.initialize_dp()

        for i in range(self._num_service):
            price_i = self._services_list[i]._price

            for j in range(self._env._budget_addition + 1):
                self.dp[i + 1][j] = self.dp[i][j]
                self.dp_record[i + 1][j]['source'] = j
                k = 0
                while True:
                    if k * price_i > j:
                        break

                    delay_reduction = 0
                    service_type = self._services_list[i]._type
                    if service_type == "a" or service_type == "b0" or service_type == "b":
                        delay_reduction = self._services_list[i].reduction_of_delay_when_add_some_server(k) * self._weight[(service_type, self._services_list[i]._id)]
                    else:
                        delay_reduction = self._services_list[i].reduction_of_delay_when_add_some_server(k) * self._weight[(service_type, (self._services_list[i]._id, self._services_list[i]._sub_id))]

                    if self.dp[i+1][j] <= self.dp[i][j - k * price_i] + delay_reduction:
                        self.dp[i+1][j] = self.dp[i][j - k * price_i] + delay_reduction
                        self.dp_record[i+1][j]['num_server'] = k
                        self.dp_record[i+1][j]['source'] = j - k * price_i
                    k += 1

        self.get_dp_result()

    def get_dp_result(self):
        i = self._num_service
        j = self._env._budget_addition


        print("service_list_len = {}, dp_record[0] = {}, dp_record[1] = {}".format(len(self._services_list), len(self.dp_record), len(self.dp_record[0])))
        while i > 0:
            print("i = {}, j = {}".format(i, j))
            print("service-{}: , type: {}, id: {}, {}".format(i, self._services_list[i-1]._type, self._services_list[i-1]._id, self._services_list[i-1]._sub_id))
            self._services_list[i-1].update_num_server(self._services_list[i-1]._num_server + self.dp_record[i][j]['num_server'])

            j = self.dp_record[i][j]['source']
            i -= 1

        self._cost = self._env.compute_cost()
        self.get_min_max_result()
        self.get_running_time()


"""
    带资源回收的贪心算法
"""
class min_max_greedy_re_allocate(Algorithm):
    def __init__(self, env):
        Algorithm.__init__(self, env)
        self._service_list = {
            "a": self._env._service_A,
            "b0": self._env._service_b0,
            "b": self._env._service_B,
            "c": self._env._service_C,
            "r": self._env._service_R
        }

        # self._is_first_set_num_server_greedy = True

    def set_num_server(self):
        self._start_time = time()
        self.get_initial_max_queuing_delay()

        while True:
            self.set_num_server_greedy()

            # 检测资源过度分配的情况
            service = self.check()
            if service is None:
                break

            # 资源回收
            # 记得把预算还回去
            s_type, s_id = service[0][0], service[0][1]
            if s_type == "a" or s_type == "b":
                self._service_list[s_type][s_id].update_num_server(self._service_list[s_type][s_id]._num_server - 1)
            else:
                self._service_list[s_type][s_id[0]][s_id[1]].update_num_server(self._service_list[s_type][s_id[0]][s_id[1]]._num_server - 1)
            self._cost = self._env.compute_cost()


            max_delay, u_i, u_j = self._env.get_max_queuing_delay_in_interaction_delay()
            print("-------------------------------------------------------")
            print("[server to reduce]: {}".format(service))
            print("[max_delay]: {}, ({}, {})".format(max_delay, u_i, u_j))
            print("[cost]: {} / {}".format(self._cost, self._env._cost_budget))

        self.get_min_max_result()
        self.get_running_time()

    def set_num_server_greedy(self):

        while self._cost < self._env._cost_budget:
            max_delay, user_i, user_j = self._env.get_max_queuing_delay_in_interaction_delay()

            user_i = self._env._users[user_i]
            user_j = self._env._users[user_j]
            service_type = ["a", "b0", "b", "c", "r"]
            service_list = [
                self._env._service_A[user_i._service_a],
                self._env._service_b0,
                self._env._service_B[user_j._service_b],
                self._env._service_C[user_j._service_c][user_j._sub_service_c],
                self._env._service_R[user_j._service_r][user_j._sub_service_r]
            ]

            services = []
            for i, s_type in enumerate(service_type):
                services.append({"service": s_type,
                                 "reduction": service_list[i].reduction_of_delay_when_add_a_server(),
                                 "price": service_list[i]._price})

            # if IF_DEBUG:
            #     a, b0, b, c, sub_c, r, sub_r = self._env.get_service_index(user_i._id, user_j._id)
            #     indices = (a, b0, b, (c, sub_c), (r, sub_r))
            #     print("service indices: {}".format(indices))
            #     print("max_delay: {}, users: ({}, {})".format(max_delay, user_i._id, user_j._id))
            #     print("services: {}".format(services))

            max_utility = self.get_max_utility(services)

            # if IF_DEBUG:
            #     print("max_utility: {}".format(max_utility))

            if max_utility is not None:
                selected_service = service_list[service_type.index(max_utility)]
                selected_service.update_num_server(selected_service._num_server + 1)

            else:
                break
            self._cost = self._env.compute_cost()

            # if IF_DEBUG:
            #     print("cost: {}, cost_budget: {}\n".format(self._cost, self._env._cost_budget))

        max_delay, u_i, u_j = self._env.get_max_queuing_delay_in_interaction_delay()
        print("-------------------------------------------------------")
        print("[max_utility]: {}".format(selected_service))
        print("[max_delay]: {}, ({}, {})".format(max_delay, u_i, u_j))
        print("[cost]: {} / {}".format(self._cost, self._env._cost_budget))


    def get_max_utility(self, services):
        services = sorted(services, key=lambda services: services["reduction"] / services["price"], reverse=True)
        max_utility = None

        for k in range(len(services)):
            if services[k]["price"] > self._env._cost_budget - self._cost:
                continue
            else:
                max_utility = services[k]["service"]
                break
        return max_utility

    """
        检查是否存在资源过度分配的问题：
        如果一个时延对不是最大时延，而且其涉及的服务减少一个服务器之后，仍然不是最大时延，那么存在资源过度分配问题。
    """

    def check(self):
        max_delay, user_i, user_j = self._env.get_max_queuing_delay_in_interaction_delay()
        # services_i_j = self._env._queuing_delay_without_duplicate[user_i][user_j]
        services_i_j = None
        for key, value in self._env._queuing_delay_without_duplicate.items():
            if value["users"] == (user_i, user_j):
                services_i_j = value["service_set"]

        """
            从除了user_i, user_j涉及的服务之外的服务中寻找
        """
        service_candidates = {}
        service_types = ["a", None, "b", "c", "r"]
        for key, value in self._env._queuing_delay_without_duplicate.items():
            user_a, user_b = value["users"][0], value["users"][1]
            if user_a == user_i and user_b == user_j:
                continue

            service_chain = value["service_set"]

            for i, service_id in enumerate(service_chain):
                if service_types[i] == "a" or service_types[i] == "b":
                    if service_id != services_i_j[i] and (self._service_list[service_types[i]][service_id]._num_server \
                            > self._env._initial_num_server["service_" + service_types[i]][service_id]):

                        imcrement = self._service_list[service_types[i]][service_id].delay_increment_when_reduce_a_server()
                        delay_after_decrease_server = self._env.compute_queuing_delay(user_a, user_b) + imcrement

                        if delay_after_decrease_server >= max_delay:
                            if (service_types[i], service_id) in service_candidates.keys():
                                # service_candidates.pop((service_types[i], service_id))
                                service_candidates[(service_types[i], service_id)] = False
                        else:
                            if (service_types[i], service_id) in service_candidates and service_candidates[(service_types[i], service_id)] is not False:
                                service_candidates[(service_types[i], service_id)] = {
                                    "increment": imcrement,
                                    "price": self._service_list[service_types[i]][service_id]._price,
                                    "num_server_after_reduce":  self._service_list[service_types[i]][service_id]._num_server - 1
                                }

                # service_id = (id, sub_id)
                elif service_types[i] == "c" or service_types[i] == "r":
                    if service_id != services_i_j[i] and (self._service_list[service_types[i]][service_id[0]][service_id[1]]._num_server \
                            > self._env._initial_num_server["service_" + service_types[i]][service_id[0]][service_id[1]]):

                        imcrement = self._service_list[service_types[i]][service_id[0]][service_id[1]].delay_increment_when_reduce_a_server()
                        delay_after_increase_server = self._env.compute_queuing_delay(user_a, user_b) + imcrement

                        if delay_after_increase_server >= max_delay:
                            if (service_types[i], service_id) in service_candidates.keys():
                                # service_candidates.pop((service_types[i], service_id))
                                service_candidates[(service_types[i], service_id)] = False
                        else:
                            if (service_types[i], service_id) in service_candidates and service_candidates[(service_types[i], service_id)] is not False:
                                service_candidates[(service_types[i], service_id)] = {
                                    "increment": imcrement,
                                    "price": self._service_list[service_types[i]][service_id[0]][service_id[1]]._price,
                                    "num_server_after_reduce": self._service_list[service_types[i]][service_id[0]][service_id[1]]._num_server - 1
                                }

                for key, value in service_candidates.items():
                    if value == False:
                        service_candidates.pop(key)

        """
            从 service_candidates 中选出一个，期望其imcrement小，且price大
        """
        if len(service_candidates) == 0:
            return None
        service_candidates = sorted(service_candidates.items(), key=lambda x: x[1]["price"] / x[1]["increment"], reverse=True)

        return service_candidates[0]