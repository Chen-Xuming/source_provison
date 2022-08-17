from numpy.random import default_rng, SeedSequence, Generator, randint
from env import *
import json

import copy
import datetime
import os
import simplejson
import psutil

cpu_no = [42]
p = psutil.Process(os.getpid())
p.cpu_affinity(cpu_no)    # 200:10
print("cpu_cores: ", cpu_no)

from algorithm import *

# import win32process
# import win32api
# win32process.SetProcessAffinityMask(win32api.GetCurrentProcess(), 0x0004)

# result_foldername = './result/2022_7_12_remove_redundant/result-{0}'.format(
#     datetime.datetime.now().microsecond)


num_instance = 1
num_service = 10
num_service_a = num_service
num_service_b = num_service
num_service_c = num_service
num_service_r = num_service


 #对比算法
#algorithm_name_list =[ 'min_max_greedy']#, 'min_max_equal_weight','min_max_surrogate_relaxation']#,'min_max_supervise', 'min_max_pulp'] #[ 'min_max_surrogate_relaxation']#, 'min_max_equal_weight', 'min_max_pulp']  'min_max_greedy'
# algorithm_name_list =[ 'min_max_greedy', 'min_max_nn']
#algorithm_name_list =[ 'min_max_surrogate_relaxation']
# algorithm_name_list =['min_max_greedy', 'min_max_equal_weight', 'min_max_surrogate_relaxation', 'min_max_pulp']

algorithm_name_list = ["min_max_equal_weight"]

# algorithm_name_list = ['min_max_pulp']

#使用固定的随机序列，随机生成的数据可重复生成
seed_sequence_list = [ 
    #SeedSequence(entropy=2192908379569140691551669790510719942),
    #SeedSequence(entropy=104106199391358918370385018247611179990),
    #SeedSequence(entropy=7350452162692297579208158309411455478),
    #SeedSequence(entropy=45880849610690604133469096444412190387),
    #SeedSequence(entropy=39934858177643291167739716489914134539),
]

'''
for n in range(num_simulation):
    if n >= len(seed_sequence_list):
        seed_sequence = SeedSequence()
    else:
        seed_sequence = seed_sequence_list[n]
    print('=====simulation : {}=====\nSeed Sequence : {}'.format(n, seed_sequence))
    rng = default_rng(seed_sequence)
    st = rng.bit_generator.state  #使不同的算法处理的数据相同
    for name in algorithm_name_list:
        print('-----algorithm : {}-----'.format(name))
        env = Env(rng, num_instance, num_service_a, num_service_b, num_service_c, num_service_r)
        algorithm_class = eval(name)
        algorithm = algorithm_class(env)
        algorithm.set_num_server()
        rng.bit_generator.state = st   #使不同的算法处理的数据相同
'''

'''
result_list = []
for n in range(1):
    seed_sequence = SeedSequence()#seed_sequence_list[1]#SeedSequence()
    print('=====simulation : {}=====\nSeed Sequence : {}'.format(n, seed_sequence))
    result = {}
    #每次仿真把预算提升
    budget_addition = 20
    for b in range(1):
        print('=====budget_addition : {}'.format(budget_addition))
        rng = default_rng(seed_sequence)
        st = rng.bit_generator.state  #使不同的算法处理的数据相同
        for name in algorithm_name_list:
            print('-----algorithm : {}-----'.format(name))
            env = Env(rng, num_instance, num_service_a, num_service_b, num_service_c, num_service_r, budget_addition)
            algorithm_class = eval(name)
            algorithm = algorithm_class(env)
            algorithm.set_num_server()
            rng.bit_generator.state = st   #使不同的算法处理的数据相同
            if name not in result:
                result[name] = []
            result[name].append(round(algorithm._queuing_delay_reduction_ratio,2))
        budget_addition += 5
    print(result)
    result_list.append(result)
print(result_list)
'''

# 读取已有实验文件中的entropy
def get_entropy(dir_path):
    entropy_list = []
    for file_name in os.listdir(dir_path):
        _, postfix = os.path.splitext(file_name)
        if postfix != '.json':
            continue
        json_file = os.path.join(dir_path, file_name)
        data = json.load(open(json_file))
        entropy = data["entropy"]
        entropy_list.append(entropy)
    return entropy_list



#min max

simulate_user_count = 400
result_dir = "result/2022-7-15-remove-redundant/equal_weight/{}u-10s".format(simulate_user_count)
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

simulate_no = 4     # simulate编号
print("simulate_no: ", simulate_no)
print("user_num: ", simulate_user_count)

entropy_files = './result/2022-7-15-remove-redundant/{}u-10s'.format(simulate_user_count)
seed_sequence_list = get_entropy(entropy_files)  # len = 50
# start_seed_no = 40
# print("seed_no: [{}, {}]".format(start_seed_no, start_seed_no + 10 - 1))

result_finished_entropy = get_entropy(result_dir)
start_no = 40
end_no = start_no + 10
seed_sequence_list = seed_sequence_list[start_no : end_no]

for n in range(10):

    if seed_sequence_list[n] in result_finished_entropy:
        continue

    # seed_sequence = SeedSequence(332209669323931633081449602077300949004)
    # seed_sequence = SeedSequence()

    # seed_sequence = SeedSequence(seed_sequence_list[n + start_seed_no])
    seed_sequence = SeedSequence(seed_sequence_list[n])

    suffix = str(seed_sequence.entropy)[-8:]
    filename_result = result_dir + '/result_random_{}.json'.format(suffix)
    
    print('=====simulation : {}=====\nSeed Sequence : {}'.format(n, seed_sequence))

    # # temp_env = Env(rng, num_instance, num_service_a, num_service_b, num_service_c, num_service_r, 10, seed_sequence)
    # print("num_group_per_instance: {}".format(temp_env._num_group_per_instance))
    # print("_num_user_per_group: {}".format(temp_env._num_user_per_group))
    # print("total_user_num: {}".format(temp_env._num_user))

    """
        如果规模env规模太大，那么初始化的时间会很长。
        对于只有budget_addition不同的env，它们只有_budget_addition, _cost_budget不同。
        可以初始化一个 budget_addition = 10 的env, 然后在不同budget_addition的实验中通过 deep_copy 拷贝，并手动改变上述两个值，这样可以大幅减少运行时间
    """
    rng = default_rng(seed_sequence)
    env_for_copy = Env(rng, num_instance, num_service_a, num_service_b, num_service_c, num_service_r, 10, seed_sequence)

    res_summary = {}
    res_summary['entropy'] = seed_sequence.entropy
   
    budget_addition = 10  #每次仿真把预算提升
    for b in range(10):
        print('=====budget : {}'.format(budget_addition))
        res_summary[budget_addition] = {}
        rng = default_rng(seed_sequence)
        st = rng.bit_generator.state  #使不同的算法处理的数据相同

        # env_for_copy = Env(rng, num_instance, num_service_a, num_service_b, num_service_c, num_service_r, budget_addition, seed_sequence)

        for name in algorithm_name_list:
            print('=====algorithm : {}====='.format(name))

            # env = Env(rng, num_instance, num_service_a, num_service_b, num_service_c, num_service_r, budget_addition, seed_sequence)
            """
                env copy & reset
            """
            env = copy.deepcopy(env_for_copy)
            env._budget_addition = budget_addition
            cost = env.compute_cost()
            env._cost_budget = cost + env._budget_addition

            res_summary[budget_addition]['num_user'] = env._num_user
            res_summary[budget_addition]['num_service'] = env._num_service
            
            algorithm_class = eval(name)
            algorithm = algorithm_class(env)
            algorithm.set_num_server()
            res_summary[budget_addition][name] = algorithm.get_result_dict()
            rng.bit_generator.state = st   #使不同的算法处理的数据相同
        budget_addition += 10

    with open(filename_result, 'a') as fid_json:            
        simplejson.dump(res_summary, fid_json)        
    
