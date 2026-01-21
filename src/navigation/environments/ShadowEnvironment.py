import copy
import json

from tinydb import TinyDB 
from openai import OpenAI

from abc import ABC, abstractmethod
from typing import Dict, Any

from navigation.environments.Environment import Environment
from utils.util import call_llm, dbToString, extract_json_from_llm_response

class ShadowEnvironment(ABC):
    def __init__(self,  hypothesesDb : TinyDB, policyDb : TinyDB, client : OpenAI, model : str):
        """
        :param use_llm_actions: Wether or not the shadow env executes actions using an LLM-based emulator.
        :param hypothesesDb: All the hypotheses about the environment.
        :param policyDb: All the strategies about the environment.
        :param client: The LLM client to use for.
        :param model: The LLM model to use.
        """
        self.client = client
        self.model = model
        self.hypothesesDb = hypothesesDb
        self.policyDb = policyDb

    def get_next_move(self, environment: Environment, sample_size=3, depth=4, use_llm_actions=False, debug=False) -> str:
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
                best_move, best_value = self.calc_best_move(list(env.ACTION_MAP.keys()), state, debug=debug)
                # 2. Add that move to trajectory
                trajectory.append((best_move, best_value))
                # 3. Execute that move in the shadow env
                response = self.execute_move(best_move, env, state, use_llm_actions=use_llm_actions)
                state = response["state"]

                if debug:
                    print(f"Sample {i + 1}, Step {j + 1}: Executed Move: {best_move}, Value: {best_value}")
                    print(f"New State:\n{state}\n")

                if(response["is_terminated"]):
                    break
            
            samples.append(trajectory)
        
        best_trajectory = self.eval_best_sample(samples)

        return best_trajectory
    
    def execute_move(self, move: str, env: Environment, llm_state: str, use_llm_actions=False, debug=False) -> Dict[str, Any]:
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
                    "is_terminated": is_terminated
                }}
                
                # Critical Instructions
                - Always respond in valid JSON format.
                - Ensure the state representation matches the format of the input state, including newline characters (\\n).
                 '''},
            ]
            
            msgStr = call_llm(self.client, self.model, prompt, debug=debug)
            if msgStr and msgStr.content:
                msg = extract_json_from_llm_response(msgStr.content)
                return msg
            else: #
                print(f"Warning: LLM shadow env returned invalid response '{msgStr}', using actual environment step.")
                return {"state": "error", "reward": 0, "is_terminated": False}
        else:
            return env.ACTION_MAP[move]()

    @abstractmethod
    def calc_best_move(self, moves: list[str], state: str) -> tuple[str, int]:
        pass

    @abstractmethod
    def eval_best_sample(self, samples: list[tuple[str, int]]) -> list[tuple[str, int]]:
        pass