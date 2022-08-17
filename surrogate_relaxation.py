import numpy as np
import sys
from time import time
from algorithm_interaction_delay import Algorithm

class min_max_surrogate_relaxation(Algorithm):
    def __init__(self, env):
        Algorithm.__init__(self, env)

        self._upper_bound = sys.maxsize
        self._lower_bound = -sys.maxsize

        self._upper_bound_list = []
        self._lower_bound_list = []

        self.best_max_delay = sys.maxsize
