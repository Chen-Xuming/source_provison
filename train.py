import numpy as np
from agent import *
from numpy.random import default_rng, SeedSequence, Generator, randint
from env import *
import datetime
import os


class Runner:
    def __init__(self, agent):
        self.agent = agent

    def step(self):
        observation = self.agent.problem_list[self.agent.problem_id].observe()
        #print('observation ',observation)
        action = self.agent.act(observation)
        (reward, done) = self.agent.problem_list[self.agent.problem_id].act(action)
        #print('[reward]: ',reward, ' [done]: ', done)
        self.agent.reward(observation, action, reward, done)
        return (observation, action, reward, done)

    def loop(self, num_problem, num_epoch, max_iter):
        cum_reward_total = 0
        for epoch in range(num_epoch):
            for p_id in range(num_problem):
                for k in range(5): #重复求解一个相同的问题5次
                    print("---epoch {}--problem {}---solve {} time".format(epoch, p_id,  k+1))
                    cum_reward = 0
                    self.agent.reset(p_id) #调用re_initialize_num_server
                    for i in range(1, max_iter + 1):
                        (obs, act, rew, done) = self.step()
                        cum_reward += rew
                        if done:
                            cum_reward_total += cum_reward
                            self.agent.wright(cum_reward_total)
                            break
                if p_id %10 == 0:
                    self.agent.save_model()


def main():
    num_instance = 1
    num_service = 5
    num_service_a = num_service
    num_service_b = num_service
    num_service_c = num_service
    num_service_r = num_service    
    budget_addition = 30
    
    seed_sequence_list = [ 
        #SeedSequence(entropy=2192908379569140691551669790510719942),
        #SeedSequence(entropy=104106199391358918370385018247611179990),
        #SeedSequence(entropy=7350452162692297579208158309411455478),
        #SeedSequence(entropy=45880849610690604133469096444412190387),
        #SeedSequence(entropy=39934858177643291167739716489914134539),
    ]    
    
    lr = 1e-4
    bs = 64 #32\64
    n_step = 1
    
    num_epoch = 10
    num_problem = 1000
    max_iter = budget_addition + 1
    problem_list = []
    for n in range(num_problem):
        seed_sequence = SeedSequence()
        suffix = str(seed_sequence.entropy)[-8:]    
        rng = default_rng(seed_sequence)
        env = Env(rng, num_instance, num_service_a, num_service_b, num_service_c, num_service_r, budget_addition, seed_sequence)
        problem_list.append(env)
    
    pattern = 'train'
    agent_class = DQAgent(problem_list, lr, bs, n_step, pattern)
    
    print("Begin training...")
    train_runner = Runner(agent_class)
    train_runner.loop(num_problem, num_epoch, max_iter)
    print("Training over")
    agent_class.save_model()
    
if __name__ == "__main__":
    main()