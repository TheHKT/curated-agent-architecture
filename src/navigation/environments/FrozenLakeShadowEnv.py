from utils.util import dbToString
import copy
import json
import random

from typing import Dict, Any

class FrozenLakeShadowEnv():
    def __init__(self,  hypothesesDb=None, strategiesDb=None, client=None, model=None):
        """
        :param use_llm_actions: Wether or not the shadow env executes actions using an LLM-based emulator.
        :param hypothesesDb: All the hypotheses about the environment.
        :param strategiesDb: All the strategies about the environment.
        :param client: The LLM client to use for.
        :param model: The LLM model to use.
        """
        self.client = client
        self.model = model
        self.hypothesesDb = hypothesesDb
        self.strategiesDb = strategiesDb
                
    def get_next_move(self, environment, sample_size=2, depth=3, use_llm_actions=False, debug=False) -> str:
        """
        This method performs lookahead simulations using the shadow environment to determine the best next move.
        
        :param environment: The actual environment from which to base the simulations.
        :param sample_size: How many sample trajectories to generate
        :param depth: How deep each trajectory should be
        :param debug: Whether to enable debug mode for detailed output
        :return: The best next move as a string e.g., "move_right"
        :rtype: str
        """
        samples = []
        for i in range(sample_size):
            env = copy.deepcopy(environment)
            state = env.get_state()
            trajectory = []
            
            for j in range(depth):
                # 1. Calc best possible move given current state
                best_move = self.calc_best_move(env.ACTION_MAP.keys(), state)
                # 2. Add that move to trajectory
                trajectory.append(best_move)
                # 3. Execute that move in the shadow env
                response = self.execute_move(best_move, env, state, use_llm_actions=use_llm_actions)
                state = response["state"]
                if(response["is_terminated"]):
                    break
            
            samples.append(trajectory)
        
        best_trajectory = self.eval_best_sample(samples)

        return best_trajectory[0]  # Return the first move of the best trajectory

    def calc_best_move(self, moves, state) -> str:
        """
        This method calculates the best possible move given the current state of the environment.
        
        :param state: The current state of the environment in a llm readable format.
        :return: The best move as a string e.g., "move_right"
        :rtype: str
        """
        # TODO: this is a placeholder
        return random.choice(list(moves))

    def eval_best_sample(self, samples):
        """
        This method evaluates which of the generated sample trajectories is the best performing and returns it.
        
        :param samples: A list of sample trajectories to evaluate.
        :return: The best performing trajectory.
        :rtype: List[str]
        """
        # TODO: this is a placeholder
        return random.choice(samples)

    def execute_move(self, move, env, llm_state, use_llm_actions=False) -> Dict[str, Any]:
        if use_llm_actions:
            prompt = [
                {"role": "system", 
                 "content": f'''
                 You are an outstanding emulator for dynamic environments. Your job is to predict the next state of the environment based on the current state, the action taken and a set of hypotheses.
                 
                 # Inputs
                 ## Current State of the Environment
                    {llm_state}
                    
                ## Hypotheses about the Environment
                    {dbToString(self.hypothesesDb)}
                    
                ## Action Taken
                    {move}
                    
                # Output
                You must output the next state of the environment in JSON format as follows:
                {{
                    "state": state,
                    "reward": reward,
                    "isTerminated": isTerminated
                }}
                
                # Critical Instructions
                - Always respond in valid JSON format.
                - Ensure the state representation matches the format of the input state, including newline characters (\\n).
                 '''},
            ]
            
            responseStr = (
            self.client.chat.completions.create(model=self.model, messages=prompt)
                .choices[0]
                .message)
            response = json.loads(responseStr.content)
            return response
        else:
            return env.ACTION_MAP[move]()
        