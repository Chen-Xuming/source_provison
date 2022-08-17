"""
    观察 min_max_greedy 运行过程
"""

from numpy.random import default_rng, SeedSequence, Generator, randint
from env import *
import json

from algorithm import min_max_greedy

num_instance = 1
num_service = 10
num_service_a = num_service
num_service_b = num_service
num_service_c = num_service
num_service_r = num_service

"""
    一些种子
    
    users = 100
        27357260251672763106192180260507998059
        131354799357392105933350882161709207040
        5646630586032672322413760579212489169
        252518865294875040193922143294314844612
        34665148710946036084855133826317892389
    
    users = 200
        156874946626639142876548184238301785633
        18723678512392186351313983442403098873
        302669776799410080675431097894204873620
        67699088520690137600914526286104930764
        
    users = 300
        229321527148788308710762291513200301017
        29181828172810716928254060530203502423
        57596961820937799310460550833103856265
        122911003508566077722123685657505380346
    
    users = 400
        71037367651468187900621311215300466576
        67492268885208061692003784061602162705
        265901610090052792399357232820102336758
        64153729983976411561310262470603917055
"""



seed_entropy = 131354799357392105933350882161709207040     # 以往实验的随机种子
print("entropy: {}\n".format(seed_entropy))

if __name__ == "__main__":
    seed_sequence = SeedSequence(seed_entropy)
    rng = default_rng(seed_sequence)

    budget_addition = 100
    server_num = 2      # 一次分配多少个服务器

    env = Env(rng, num_instance, num_service_a, num_service_b, num_service_c, num_service_r, budget_addition, seed_sequence)
    print("num_group_per_instance: {}".format(env._num_group_per_instance))
    print("_num_user_per_group: {}".format(env._num_user_per_group))
    print("total_user_num: {}".format(env._num_user))
    print("budget: {}".format(budget_addition))
    print("server_num: {}".format(server_num))

    algorithm = min_max_greedy(env, server_num)
    algorithm.set_num_server()

