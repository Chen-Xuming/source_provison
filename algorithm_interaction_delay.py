"""
    algorithm2.py 考虑优化整个交互时延的版本
    交互时延 = 传输时延 + 排队时延 + 处理时延
"""

import numpy as np
import sys
from time import time
import pulp
from env2 import *
import simplejson

class Algorithm:
    def __init__(self, env):
        self._env = env
        self._max_interaction_delay = 0     # 最大交互时延

        self._start_time = None
        self._end_time = None

        self.result_dict = {}

    def get_running_time(self):
        self._end_time = time()
        self._running_time = (self._end_time - self._start_time) * 1000  # ms

    def set_num_server(self):
        pass

    # min max
    def get_initial_max_interaction_delay(self):
        self._cost = self._env.compute_cost()
        max_delay, user_i, user_j = self._env.get_max_interaction_delay()
        self._max_interaction_delay_initial = max_delay
        print('initial max interaction delay: [', self._max_interaction_delay_initial, '] cost ', self._cost, ' budget ',
              self._env._cost_budget)

    # min max
    def get_min_max_result(self):
        cost = self._env.compute_cost()
        max_delay, user_i, user_j = self._env.get_max_interaction_delay()
        self._max_interaction_delay = max_delay
        print('final max interaction delay [', self._max_interaction_delay, '] cost ', self._cost, ' budget ',
              self._env._cost_budget)
        assert cost <= self._env._cost_budget

    def get_result_dict(self):
        # self.result_dict = {'max_delay': self._max_interaction_delay, 'running_time': self._running_time}
        self.result_dict["max_delay"] = self._max_interaction_delay
        self.result_dict["running_time"] = self._running_time
        return self.result_dict

IF_DEBUG = False


class min_max_greedy(Algorithm):
    def __init__(self, env):
        Algorithm.__init__(self, env)

    def set_num_server(self):
        self._solutions = []

        self._start_time = time()
        self.get_initial_max_interaction_delay()

        while self._cost < self._env._cost_budget:
            max_delay, user_i, user_j = self._env.get_max_interaction_delay()

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
                # 这里计算的增加一台服务器减少的时延量，与优化目标考虑的是整个时延还是排队时延没有关系。
                services.append({"service": s_type,
                                 "reduction": service_list[i].reduction_of_delay_when_add_a_server(),
                                 "price": service_list[i]._price,
                                 "id": (service_list[i]._id, service_list[i]._sub_id)})

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

                a, b0, b, c, sub_c, r, sub_r = self._env.get_service_index(user_i._id, user_j._id)
                indices = (a, b0, b, (c, sub_c), (r, sub_r))
                self._solutions.append([max_utility, indices[service_type.index(max_utility)], max_delay, (user_i._id, user_j._id)])
            else:
                break
            self._cost = self._env.compute_cost()


            max_delay, user_i, user_j = self._env.get_max_interaction_delay()
            print("max_delay = {}".format(max_delay))

            if IF_DEBUG:
                print("cost: {}, cost_budget: {}\n".format(self._cost, self._env._cost_budget))

        self.get_min_max_result()
        self.get_running_time()

        # --------- solutions -----------------
        for solution in self._solutions:
            print("({}, {}),  max_delay = {}, user_pair:{}".format(solution[0], solution[1], solution[2], solution[3]))

    def get_max_utility(self, services):
        services = sorted(services, key=lambda services: services["reduction"] / services["price"], reverse=True)
        max_utility = None

        for k in range(len(services)):
            if services[k]["price"] > self._env._cost_budget - self._cost:
                continue
            else:
                max_utility = services[k]["service"]

                if max_utility:
                    print("service_to_add: {}".format(services[k]))

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
        # global service
        self._start_time = time()
        self.get_initial_max_interaction_delay()

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

                # 每一对用户的交互时延都要小于M
                model += self._env.compute_interaction_delay(user_i, user_j) - \
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

        # solver = pulp.PULP_CBC_CMD(msg=True, warmStart=False, timeLimit=1800)  # timeLimit=600 ===> 10min
        solver = pulp.PULP_CBC_CMD(msg=False, warmStart=False)

        print("Pulp solving......")
        model.solve(solver)

        solution_status = pulp.LpStatus[model.status]
        print("Status of pulp's solution: {}".format(solution_status))

        self.result_dict["pulp_solution_status"] = solution_status

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
        # print("\n------------- Info -------------")
        # print("\n------------ services -----------------")
        # for key, service_ in self._services_dict.items():
        #     print("{}: {}".format(key, service_._id))
        # print("\n------------ reductions ----------------")
        # for key, reductions in self._delay_reductions_each_service.items():
        #     print("{}: {}".format(key, reductions))
        # print("\n------------ x -------------------")
        # for key, value in x.items():
        #     # print("{}: {}".format(key, value))
        #     # for x_ in value:
        #     #     if x_.value() == 1:
        #     #         print("num_server: {}".format(x_))
        #     for x_ in value:
        #         if x_.value() == 1:
        #             num_server = int(x_.name.split("_")[-1])
        #             if num_server > 0:
        #                 print("{}: {}".format(key, num_server))


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
        self._solutions = []

        self._start_time = time()
        self.get_initial_max_interaction_delay()
        self.compute_weight()
        while self._cost < self._env._cost_budget:
            services = []
            for key in self._weight.keys(): # key = (type, id)
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
                    self._service_list[s_type][s_id[0]][s_id[1]].update_num_server(self._service_list[s_type][s_id[0]][s_id[1]]._num_server + 1)

                self._solutions.append([s_type, s_id])

            else:
                break
            self._cost = self._env.compute_cost()

        self.get_min_max_result()
        self.get_running_time()

        # --------- solutions -----------------
        for solution in self._solutions:
            print("({}, {})".format(solution[0], solution[1]))


    def compute_weight(self):
        numerator = len(self._env._interaction_delay_without_duplicate) ** 2
        self._weight = {}

        service_types = ["a", "b0", "b", "c", "r"]
        for key, value in self._env._interaction_delay_without_duplicate.items():
            for i, service_id in enumerate(value["service_chain"]):
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
        self._initial_upper_bound = self._max_interaction_delay
        self._start_time = time()

        self.surrogate()

        # """
        #     不断尝试a值，获取最好的解
        # """
        # print("-----------")
        # high_a = 3
        # low_a = 0
        # mid = 0
        # convergence_a = -1
        # convergence_max_interaction_delay = -1
        # while high_a - low_a > 0.05:
        #     mid = (low_a + high_a) / 2
        #     # print("-----------")
        #     print("mid = {}".format(mid))
        #     if self.surrogate(mid):
        #         high_a = mid
        #         convergence_a = mid
        #         convergence_max_interaction_delay = self._max_interaction_delay
        #         # print("convergence")
        #     else:
        #         low_a = mid
        #         # print("can't convergence")
        # # print("Final a = {}".format(mid))
        # print("Convergence a = {}".format(convergence_a))
        # # self.surrogate(a=convergence_a)
        # self._max_interaction_delay = convergence_max_interaction_delay


        self.get_running_time()

        self._upper_bound = min(self._upper_bound_list)
        self._lower_bound = max(self._lower_bound_list)
        assert self.best_max_delay == self._upper_bound
        # print('upper_bound = ', self._upper_bound)
        # print('lower_bound = ', self._lower_bound)

        # --------- solutions -----------------
        # for solution in self._solutions:
        #     print("({}, {})".format(solution[0], solution[1]))
        for constraint in self._constraints:
            print(constraint)
        # print("amount of constraint: {}/{}".format(len(self._constraints), len(self._env._interaction_delay_without_duplicate)))
        # print("UB_list: {}".format(self._upper_bound_list))
        # print("LB_list: {}".format(self._lower_bound_list))

        max_delay, user_i, user_j = self._env.get_max_interaction_delay()
        print("max_delay = ", max_delay)

    def set_num_server_min_max_greedy(self):
        self.get_initial_max_interaction_delay()
        while self._cost < self._env._cost_budget:
            max_delay, user_i, user_j = self._env.get_max_interaction_delay()
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
    def get_initial_max_interaction_delay(self):
        self._cost = self._env.compute_cost()
        max_delay, user_i, user_j = self._env.get_max_interaction_delay()
        self._max_interaction_delay_initial = max_delay
        print('initial max interaction delay: [', self._max_interaction_delay_initial, '] cost ', self._cost, ' budget ',
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
            print("constraints: ---------------------")
            for constraint in self._constraints:
                print(constraint)
            # print("amount of constraint: {}/{}".format(len(self._constraints),
            #                                            len(self._env._interaction_delay_without_duplicate)))

            f_lambda_k = self.line_search()     # lower bound

            max_delay, user_i, user_j = self._env.get_max_interaction_delay()
            print("max_delay = {}, users = ({}, {})".format(max_delay, user_i, user_j))


            print("f_lambda_k: {:.7f}, f_lambda: {}, self._initial_upper_bound * delta: {}".format(f_lambda_k, f_lambda, self._initial_upper_bound * delta))
            if f_lambda_k - f_lambda > self._initial_upper_bound * delta:
            # if max_delay > self._initial_upper_bound:
            # if (max_delay - f_lambda_k) / f_lambda_k > 0.60:
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

        return -1   # 多余

    # """
    #     判断是否收敛：
    #     如果constraint个数超过 1/4 的时延总数，就认为给定的a值下不可以收敛。
    # """
    # def surrogate(self, a):
    #     # print("----------------")
    #     # print("a = {}".format(a))
    #     convergence = True
    #
    #     delta = 0.00001
    #     self.select_two_max_constraint()
    #     k = 1
    #     f_lambda = - sys.maxsize
    #     while True:
    #         if len(self._constraints) > len(self._env._interaction_delay_without_duplicate) / 5:
    #             convergence = False
    #             break
    #
    #         # print("constraints: ---------------------")
    #         # for constraint in self._constraints:
    #         #     print(constraint)
    #         # print("amount of constraint: {}/{}".format(len(self._constraints),
    #         #                                            len(self._env._interaction_delay_without_duplicate)))
    #
    #         f_lambda_k = self.line_search()     # lower bound
    #
    #         max_delay, user_i, user_j = self._env.get_max_interaction_delay()
    #         # print("max_delay = {}".format(max_delay))
    #
    #
    #         # print("f_lambda_k: {:.7f}, f_lambda: {}, self._initial_upper_bound * delta: {}".format(f_lambda_k, f_lambda, self._initial_upper_bound * delta))
    #         # if f_lambda_k - f_lambda > self._initial_upper_bound * delta:
    #         # if max_delay > self._initial_upper_bound:
    #         if (max_delay - f_lambda_k) / f_lambda_k > a:
    #             self._constraints.append(
    #                 {
    #                     'users': (user_i, user_j),
    #                     'lambda': 1,
    #                     'a': self._env._users[user_i]._service_a,
    #                     'b0': 0,
    #                     'b': self._env._users[user_j]._service_b,
    #                     'c': (self._env._users[user_j]._service_c, self._env._users[user_j]._sub_service_c),
    #                     'r': (self._env._users[user_j]._service_r, self._env._users[user_j]._sub_service_r)
    #                 }
    #             )
    #             f_lambda = f_lambda_k
    #             k += 1
    #         else:
    #             f_lambda = f_lambda_k
    #             # return f_lambda
    #             break
    #
    #     return convergence


    def select_two_max_constraint(self):
        self._env.re_initialize_num_server()

        # 保存约束（一个客户端对的排队时延）的权重、其中包含的服务
        self._constraints = []
        max_delay, _2nd_max_delay = self._env.get_top_two_interaction_delay()

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

    """
        寻找一个权重lambda, 使得lower bound最大
    """
    def line_search(self):
        print("---------------- line search ------------------------------------------------------")

        epsilon = 0.05  # 0.05
        delta = 0.05    # 0.05
        lambda_low = 0.
        lambda_high = 1.
        k = 1
        f_lambda = 0
        lambda_k = 0

        best_lambda_k = -1
        max_low_bound = -1
        while True:
            if lambda_high - lambda_low < epsilon:
                # print("iter_count = {}".format(iter_count))
                break
            lambda_k = lambda_low + (lambda_high - lambda_low) / 2

            # print("----------- lambda line search ------------")
            print("lambda_low: {}, lambda_high: {}, lambda_k: {}".format(lambda_low, lambda_high, lambda_k))

            # f_lambda_delta = self.compute_min_sum_allocation_solution(lambda_k + delta)
            f_lambda_delta = self.compute_min_sum_allocation_solution(min(lambda_k + delta, 1))     # + delta 之后可能大于1
            f_lambda = self.compute_min_sum_allocation_solution(lambda_k)

            print("f_lambda = {}, f_lambda_delta = {}".format(f_lambda, f_lambda_delta))

            if f_lambda > max_low_bound:
                max_low_bound = f_lambda
                best_lambda_k = lambda_k

            self._lower_bound_list.append(f_lambda)
            self._upper_bound_list.append(self._max_interaction_delay)

            # print("----------- bounds ------------")
            print("LB = {}, UB = {}".format(f_lambda, self._max_interaction_delay))


            if self.best_max_delay > self._max_interaction_delay:
                self.best_max_delay = self._max_interaction_delay
                # self.save_constraint_weight(lambda_k)

            if (f_lambda_delta - f_lambda) / delta <= 0:
                lambda_high = lambda_k
            else:
                lambda_low = lambda_k
            k += 1

        """
            用最佳的lambda进行分配
        """
        print("best lambda = {}".format(best_lambda_k))
        lambda_k = best_lambda_k
        self.compute_min_sum_allocation_solution(lambda_k, debug=True)

        w = 0
        for i in range(len(self._constraints)):
            if w == len(self._constraints) - 1:
                self._constraints[i]['lambda'] *= (1 - lambda_k)
            else:
                self._constraints[i]['lambda'] *= lambda_k
            w += 1

        return f_lambda

        # 用线性搜索找最大 lower_bound 及对应 lambda_k
        # lambda_k = 0
        # max_lower_bound = -1
        # best_lambda_k = -1
        # while lambda_k < 1.0:
        #     print("lambda_k = {}".format(lambda_k))
        #     f_lambda = self.compute_min_sum_allocation_solution(lambda_k)
        #
        #     self._lower_bound_list.append(f_lambda)
        #     self._upper_bound_list.append(self._max_interaction_delay)
        #
        #     if f_lambda > max_lower_bound:
        #         max_lower_bound = f_lambda
        #         best_lambda_k = lambda_k
        #
        #     if self.best_max_delay > self._max_interaction_delay:
        #         self.best_max_delay = self._max_interaction_delay
        #
        #     print("LB = {}, UB = {}".format(f_lambda, self._max_interaction_delay))
        #     lambda_k += 0.05
        #
        # print("best lambda_k = {}".format(best_lambda_k))
        # print("max_lower_bound = {}".format(max_lower_bound))
        # self.compute_min_sum_allocation_solution(best_lambda_k)     # 最优解
        # w = 0
        # for i in range(len(self._constraints)):
        #     if w == len(self._constraints) - 1:
        #         self._constraints[i]['lambda'] *= (1 - lambda_k)
        #     else:
        #         self._constraints[i]['lambda'] *= lambda_k
        #     w += 1
        #
        # return f_lambda


    def get_services_for_utility(self, lambda_k):
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
                                                                     constraint['c'][1]].reduction_of_delay_when_add_a_server() * constraint['lambda'] * weight
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
        return services



    def compute_min_sum_allocation_solution(self, lambda_k, debug=False):
        self._solutions = []    # 把解记录下来（debug）

        self._env.re_initialize_num_server()
        self._cost = self._env.compute_cost()

        service_over_budget = None
        while self._cost < self._env._cost_budget:
            services = self.get_services_for_utility(lambda_k)

            """
                情况1：max_utility非None而且不是第一个，那么service_over_budget也不是None，为要找的终止条件的非可行解
                情况2：max_utility是第一个，且预算刚好用完，那么service_over_budget=None, 此时需要再运行一次get_max_utility并直接返回第一个
                情况3：max_utility = None, 那么service_over_budget非None, 此时需要再运行一次get_max_utility并直接返回第一个
                
                总结：只要service_over_budget是None，就直接获取utility最大的
            """
            service_over_budget, max_utility = self.get_max_utility_min_sum(services)

            if debug:
                print("------------------------")
                print("service_over_budget = {}".format(service_over_budget))
                print("max_utility = {{type: {}, index: {}, reduction: {:.7f}, price: {}}}".format(max_utility["type"], max_utility["index"],
                                                                                                 max_utility["reduction"], max_utility["price"]))


            if max_utility:
                if max_utility['type'] == 'a':
                    max_utility = self._env._service_A[max_utility['index']]
                    # self._env._service_A[max_utility['index']].update_num_server(self._env._service_A[max_utility['index']]._num_server + 1)
                elif max_utility['type'] == 'b0':
                    max_utility = self._env._service_b0
                    # self._env._service_b0.update_num_server(self._env._service_b0._num_server + 1)
                elif max_utility['type'] == 'b':
                    max_utility = self._env._service_B[max_utility['index']]
                    # self._env._service_B[max_utility['index']].update_num_server(self._env._service_B[max_utility['index']]._num_server + 1)
                elif max_utility['type'] == 'c':
                    max_utility = self._env._service_C[max_utility['index'][0]][max_utility['index'][1]]
                    # self._env._service_C[max_utility['index'][0]][max_utility['index'][1]].update_num_server(self._env._service_C[max_utility['index'][0]][max_utility['index'][1]]._num_server + 1)
                elif max_utility['type'] == 'r':
                    max_utility = self._env._service_R[max_utility['index'][0]][max_utility['index'][1]]
                    # self._env._service_R[max_utility['index'][0]][max_utility['index'][1]].update_num_server(self._env._service_R[max_utility['index'][0]][max_utility['index'][1]]._num_server + 1)

                self._solutions.append([max_utility._type, (max_utility._id, max_utility._sub_id)])

                max_utility.update_num_server(max_utility._num_server + 1)

            else:
                break

            self._cost = self._env.compute_cost()

        cost = self._env.compute_cost()
        max_delay, user_i, user_j = self._env.get_max_interaction_delay()
        self._max_interaction_delay = max_delay     # 这个是可行解
        assert cost <= self._env._cost_budget

        if max_utility:
            max_utility.update_num_server(max_utility._num_server - 1)



        """
            计算不可行解对应的 f(λ) 值
            上面service_over_budget记录的是超出预算时的服务，我们需要假装给他增加一个服务器,计算f并返回。
        """
        # print("------------ get_over_budget --------------")
        # print("cost = {}, cost_budget = {}".format(cost, self._env._cost_budget))
        if service_over_budget:
            pass
        else:
            services = self.get_services_for_utility(lambda_k)
            value_list = []
            for v in services.values():
                value_list.append(v)
            value_list = sorted(value_list, key=lambda value_list: value_list['reduction'] / value_list['price'], reverse=True)
            service_over_budget = value_list[0]

        if service_over_budget["type"] == "a":
            service_over_budget = self._env._service_A[service_over_budget["index"]]
        elif service_over_budget["type"] == "b0":
            service_over_budget = self._env._service_b0
        elif service_over_budget["type"] == "b":
            service_over_budget = self._env._service_B[service_over_budget["index"]]
        elif service_over_budget["type"] == "c":
            service_over_budget = self._env._service_C[service_over_budget["index"][0]][service_over_budget["index"][1]]
        elif service_over_budget["type"] == "r":
            service_over_budget = self._env._service_R[service_over_budget["index"][0]][service_over_budget["index"][1]]

        service_over_budget.update_num_server(service_over_budget._num_server + 1)

        # print("service_over_budget price: {}".format(service_over_budget._price))


        f_lambda_infeasible = 0
        w = 0
        for constraint in self._constraints:
            weight = lambda_k
            if w == len(self._constraints) - 1:
                weight = 1 - lambda_k
            f_lambda_infeasible += self._env.compute_interaction_delay(constraint['users'][0], constraint['users'][1]) * constraint['lambda'] * weight
            w += 1

        # 将服务器还回去
        service_over_budget.update_num_server(service_over_budget._num_server - 1)

        # 把max_utility的服务器加回来
        if max_utility:
            max_utility.update_num_server(max_utility._num_server + 1)

        return f_lambda_infeasible

    def get_max_utility_min_sum(self, services):
        # is_last = False     # 是否是最后一次调用

        value_list = []
        for v in services.values():
            value_list.append(v)
        value_list = sorted(value_list, key=lambda value_list: value_list['reduction'] / value_list['price'],
                            reverse=True)

        max_utility = None
        service_over_budget = None      # 超出预算的那个服务
        for k in range(len(value_list)):
            if value_list[k]['price'] > self._env._cost_budget - self._cost:

                # print("service over budget....... = {}".format(service_over_budget["index"]))
                # 记录下第一个超出预算的服务
                if service_over_budget is None:
                    service_over_budget = value_list[k]
                    # print("service over budget = {}".format(service_over_budget["index"]))

                continue
            else:
                max_utility = value_list[k]
                # print("max_utility = {}".format(max_utility["index"]))
                break

        #

        return service_over_budget, max_utility


    # def get_max_utility_min_sum(self, services):
    #     value_list = []
    #     for v in services.values():
    #         value_list.append(v)
    #     value_list = sorted(value_list, key=lambda value_list: value_list['reduction'] / value_list['price'],
    #                         reverse=True)
    #
    #     max_utility = None
    #     for k in range(len(value_list)):
    #         if value_list[k]['price'] > self._env._cost_budget - self._cost:
    #             continue
    #         else:
    #             max_utility = value_list[k]
    #             break
    #     return max_utility


