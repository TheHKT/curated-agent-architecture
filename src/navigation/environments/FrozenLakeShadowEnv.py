from utils.util import call_llm, dbToString, extract_json_from_llm_response
import random
from navigation.environments.ShadowEnvironment import ShadowEnvironment

class FrozenLakeShadowEnv(ShadowEnvironment):            
    def calc_best_move(self, moves : list[str], state: str, debug=False) -> tuple[str, int]:
        """
        This method calculates the best possible move given the current state of the environment.
        
        :param state: The current state of the environment in a llm readable format.
        :return: The best move as a string e.g., "move_right"
        :rtype: str
        """
        prompt = [
        {
            "role": "system",
            "content": f"""
            You are a value estimator for a 2D navigation agent in a dynamic grid environment.

            # Agent Goal
            The agents needs to navigate from start (S) to goal (G). Current position is marked with [ ].

            # Your Task
            Rate each available move (0-100) based on how well it helps reach the goal while considering the given policies and hypotheses.

            # Inputs
            ## Environment Hypotheses
            These are beliefs about how the environment itself operates:
            {dbToString(self.hypothesesDb)}

            ## Proven Policies
            These are policies, common mistakes or guidance to help you rate the moves:
            {dbToString(self.policyDb)}

            ## Current State
            {state}

            ## Available Moves
            {', '.join(moves)}

            # Rating Process
            1. Identify current position [ ] and goal G in the state
            2. For each move, consider:
               - Does it move toward or away from G?
               - Does the policy guidance suggest rating it high or low?
               - How does the hypotheses about the environment impact the move's effectiveness and rating?
            3. Assign value (0-100):
               * 0-20: Dangerous or counterproductive
               * 21-40: Poor progress toward goal
               * 41-60: Neutral or sideways movement
               * 61-80: Good progress toward goal
               * 81-100: Optimal progress, aligns with all policy guidance

            # Output Format
            Return ONLY the best move and its value in JSON format with NO additional text:
            {{"move": "move_1", "value": 67}}

            - move: Exactly one move from Available Moves list (case-sensitive)
            - value: Integer 0-100

            CRITICAL: Output ONLY the JSON object. No explanations, markdown, or extra text.   
            """,
        }
    ]
        msg = call_llm(self.client, self.model, prompt, debug=debug)
        if msg and msg.content:
            best_move = extract_json_from_llm_response(msg.content)
            if best_move is not None and "move" in best_move and "value" in best_move and best_move["move"] in moves:
                return (best_move["move"], best_move["value"])
            else:
                print(f"Warning: LLM value-function returned invalid response '{best_move}', using random.")
                return (random.choice(moves), 50)        

    def eval_best_sample(self, samples: list[tuple[str, int]]) -> list[tuple[str, int]]:
        """
        This method evaluates which of the generated sample trajectories is the best performing and returns it.
        
        :param samples: A list of sample trajectories to evaluate.
        :return: The best performing trajectory.
        :rtype: List[str]
        """
        gamma = 0.9
    
        def discounted_value(trajectory):
            return sum(gamma**i * value for i, (_, value) in enumerate(trajectory))
    
        best_trajectory = max(samples, key=discounted_value)
        return best_trajectory