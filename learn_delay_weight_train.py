import numpy as np
from learn_delay_weight_agent import *
from numpy.random import default_rng, SeedSequence, Generator, randint
from env import *
import datetime
import os


class Runner:
    def __init__(self, agent):
        self.agent = agent
    '''
    def step(self):
        observation = self.agent.problem_list[self.agent.problem_id].observe()
        action = self.agent.act(observation)
        (reward, done) = self.agent.problem_list[self.agent.problem_id].act(action)
        self.agent.reward(observation, action, reward, done)
        return (observation, action, reward, done)
    '''
    def loop(self, num_problem, num_epoch):
        cum_reward_total = 0
        for epoch in range(num_epoch):
            for p_id in range(num_problem):
                print("---epoch {}--problem {}".format(epoch, p_id))
                cum_reward = 0
                self.agent.reset(p_id) 
                self.agent.learn_delay_weight()
                '''
                for i in range(1, max_iter + 1):
                    (obs, act, rew, done) = self.step()
                    cum_reward += rew
                    if done:
                        cum_reward_total += cum_reward
                        self.agent.wright(cum_reward_total)
                        break
                '''
                if p_id %10 == 0:
                    self.agent.save_model()


def main():
    num_instance = 1
    num_service = 5
    num_service_a = num_service
    num_service_b = num_service
    num_service_c = num_service
    num_service_r = num_service    
    budget_addition = 50
    
    
    lr = 1e-4
    bs = 128 #32\64
    n_step = 1
    
    num_epoch = 5
    num_problem = 1000
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
    train_runner.loop(num_problem, num_epoch)
    print("Training over")
    agent_class.save_model()
    
if __name__ == "__main__":
    main()