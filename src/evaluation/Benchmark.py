from openai import OpenAI
from tinydb import TinyDB
import pandas as pd
import os
from navigation.Navigator import Navigator
from navigation.environments.Environment import Environment
from navigation.environments.ShadowEnvironment import ShadowEnvironment
from optimization.hypotheses.HypothesesRefiner import HypothesesRefiner
from optimization.policy.PolicyRefiner import PolicyRefiner
from optimization.prompts import Prompts


class Benchmark:
    
    def __init__(self, env:  Environment, shadow_env: ShadowEnvironment, policyDb: TinyDB, hypothesesDb: TinyDB, optimizationPrompts: Prompts, client: OpenAI, model: str):
        self.env = env
        self.shadow_env = shadow_env
        self.policyDb = policyDb
        self.hypothesesDb = hypothesesDb
        self.optimizationPrompts = optimizationPrompts
        self.client = client
        self.model = model
        
    def run(self, name, iteration_depth, lookahead_depth, lookahead_sample_size, use_llm_action, max_nav_steps, print_debug=False):
        raw_df = []
        try:
            for i in range(iteration_depth):
                try:
                    self.env.reset()
                    navigator = Navigator(self.env, self.shadow_env)
                    trajectory, raw_metrics = navigator.run(lookahead_sample_size, lookahead_depth, max_nav_steps, use_llm_action, debug=print_debug)

                    raw_df.extend(self.transform_metrics(i, raw_metrics))

                    # Optimization loop to refine hypotheses and strategies based on trajectory
                    hypothesisRefiner = HypothesesRefiner(self.client, self.model, self.hypothesesDb)
                    hypothesisRefiner.run(trajectory, debug=print_debug)

                    policyRefiner = PolicyRefiner(self.client, self.model, self.policyDb, self.optimizationPrompts)
                    policyRefiner.run(trajectory, debug=print_debug)
                except Exception as e:
                    if print_debug:
                        print(f"Exception in iteration {i}: {e}; skipping")
                    continue
        finally:
            df = pd.DataFrame(raw_df)
            filename = f'{name}_iterations={iteration_depth}_lookaheaddepth={lookahead_depth}_lookaheadsize={lookahead_sample_size}_llmaction={use_llm_action}'
            full_path = f'../results/{filename}.csv'
            df.to_csv(full_path, index=False)
            
            self.store_dbs(filename)
        
            return df
    
    def transform_metrics(self, iteration, raw_metrics):
        metrics = []
        for step_idx, step in enumerate(raw_metrics):
            metrics.append({
                'iteration': iteration,
                'step': step_idx,
                'reward': step['reward'],
                'is_terminated': step['is_terminated']
            })
            
        return metrics
    
    def store_dbs(self, filename):
        policy_path = f"../results/policies/{filename}.json"
        hypotheses_path = f"../results/hypotheses/{filename}.json"
        
        if(os.path.exists(policy_path)):
            os.remove(policy_path)
        if(os.path.exists(hypotheses_path)):
            os.remove(hypotheses_path)
        
        new_policyDb = TinyDB(policy_path)
        new_hypothesesDb = TinyDB(hypotheses_path)  
        
        for item in self.policyDb.all():
            new_policyDb.insert(item)
        for item in self.hypothesesDb.all():
            new_hypothesesDb.insert(item)