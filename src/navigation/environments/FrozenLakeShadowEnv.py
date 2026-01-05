from utils.util import dbToString
import copy
import json
import random
from navigation.environments.ShadowEnvironment import ShadowEnvironment
from typing import Dict, Any

class FrozenLakeShadowEnv(ShadowEnvironment):            
    def calc_best_move(self, moves : list, state: str) -> str:
        """
        This method calculates the best possible move given the current state of the environment.
        
        :param state: The current state of the environment in a llm readable format.
        :return: The best move as a string e.g., "move_right"
        :rtype: str
        """
        # TODO: this is a placeholder
        return random.choice(moves)

    def eval_best_sample(self, samples: list) -> list:
        """
        This method evaluates which of the generated sample trajectories is the best performing and returns it.
        
        :param samples: A list of sample trajectories to evaluate.
        :return: The best performing trajectory.
        :rtype: List[str]
        """
        # TODO: this is a placeholder
        return random.choice(samples)