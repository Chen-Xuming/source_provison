from env2 import *
from numpy.random import default_rng, SeedSequence, Generator, randint
from algorithm_interaction_delay import *

# from algorithm2 import *

num_instance = 1
num_service = 10

env_config = {
    "num_instance": num_instance,
    "num_service_a": num_service,
    "num_service_b": num_service,
    "num_service_c": num_service,
    "num_service_r": num_service,
    "budget_addition": 100,
    "num_group_per_instance": 10,
    "num_user_per_group": 10,
    "min_arrival_rate": 50,
    "max_arrival_rate": 60,
    "min_service_rate": 500,
    "max_service_rate": 600,
    "min_price": 1,
    "max_price": 5,
    "trigger_probability": 0.2,
    "min_tx": 5,       # 传输时延
    "max_tx": 10,
}

entropy = 256298045951173193834638297830043919441     # delay = 5s
# entropy = 268359714054752458600856382210601121357
seed_sequence = SeedSequence(entropy)
# seed_sequence = SeedSequence()
rng = default_rng(seed_sequence)
print("seed = {}".format(seed_sequence.entropy))

env = Env(env_config, rng, seed_sequence)

# print(" --------------- service a -----------------")
# for service in env._service_A:
#     print(service)
# print("\n ---------------- service b0 ------------------")
# print(env._service_b0)
# print(" --------------- service b -----------------")
# for service in env._service_B:
#     print(service)
# print(" --------------- service c -----------------")
# for i in range(env._num_service_c):
#     for service in env._service_C[i]:
#         print(service)
# print(" --------------- service r -----------------")
# for i in range(env._num_service_r):
#     for service in env._service_R[i]:
#         print(service)

# service_ = None
# for i in range(env._num_service_c):
#     for service in env._service_C[i]:
#         if service._type == "c" and service._id == 7 and service._sub_id == 2:
#             service_ = service
#             break
#     if service_:
#         break

# service_.compute_queuing_delay(service_._num_server)
# print(service_.compute_queuing_delay_2(3))


# env.check_users()
# env.check_services()
# for key, value in env._interaction_delay_without_duplicate.items():
#     print("{}: {}".format(key, value))

# for key, value in env._queuing_delay_without_duplicate.items():
#     print("{}: {}".format(key, value))

# print("\n\n")
#
# alg = min_max_greedy(env)
# alg = min_max_equal_weight_dp(env)

# alg = min_max_equal_weight(env)
alg = min_max_surrogate_relaxation(env)
# alg = min_max_pulp(env)

# alg = min_max_greedy_re_allocate(env)


alg.set_num_server()
print("")
print(alg._running_time)
# print(alg._max_delay)
print(alg._max_interaction_delay)

# 14-02


