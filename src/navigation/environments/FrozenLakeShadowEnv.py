from utils.util import call_llm, dbToString, extract_json_from_llm_response
import random
from navigation.environments.ShadowEnvironment import ShadowEnvironment

class FrozenLakeShadowEnv(ShadowEnvironment):            
    def calc_best_move(self, moves : list[str], state: str, lookahead_trajectory: str, previous_trajectory: str, debug=False) -> tuple[str, int]:
        """
        This method leverages an LLM to evaluate and rate possible moves from the current state, considering the environment's hypotheses and policies.
        
        :param moves: Available moves in the environment.
        :type moves: list[str]
        :param state: The current state of the environment.
        :type state: str
        :param lookahead_trajectory: The trajectory of moves considered in the lookahead search.
        :type lookahead_trajectory: str
        :param previous_trajectory: The trajectory of moves executed so far.
        :type previous_trajectory: str
        :param debug: Whether to enable debug mode for detailed output.
        :type debug: bool
        :return: The best move and its value as a tuple.
        :rtype: tuple[str, int]
        """
        prompt = [
        {
            "role": "system",
            "content": 
f"""
You are a value estimator for a 2D navigation agent in a dynamic grid environment. You are used as a value function to rate possible moves for a lookahead search.
Those moves are executed in a shadow environment, which is a simulator of the actual environment used to perform the lookahead search and evaluate the possible trajectories.

# Your Task
Rate each available move (0-100) based on how well it helps reach the goal while considering the given strategies proofen for the environment, hypotheses about the environment and the previous moves performed by the lookahead search.

# Inputs
## Navigation Agent Goal:
The navigation agent needs to navigate from start (S) to goal (G). Current position is marked with [ ].

## Environment Hypotheses - Beliefs about how the environment itself operates:
{dbToString(self.hypothesesDb) if self.hypothesesDb is not None and dbToString(self.hypothesesDb) else 'No hypotheses available yet!'}

## Proven Strategies - Policies, Guidance, and Common Mistakes to help you rate the moves:
{dbToString(self.policyDb) if self.policyDb is not None and dbToString(self.policyDb) else 'No strategies available yet!'}

## Previous Trajectory - The moves executed in the actual environment so far; the last state in there is the startpoint for the lookahead search in the shadow environment:
{previous_trajectory}

## Previous Moves - The moves executed in the Shadow Environment so far in this lookahead search:
{lookahead_trajectory}

## Current State of the Shadow Environment:
{state}

## Available Moves - Possible actions the agent can take from the current state:
{', '.join(moves)}

# Rating Process
1. Identify current position [ ] and goal G in the Current State of the Shadow Environment
2. For each move, consider:
   - Does it move toward or away from G?
   - Does the provided strategies suggest rating it high or low?
   - How do the hypotheses about the environment dynamics affect the move's effectiveness and rating?
   - How do the previous moves influence the rating of this move? Does it show a pattern of progress or getting stuck?
3. Assign value (0-100):
   * 0-20: Dangerous or counterproductive
   * 21-40: Poor progress toward goal
   * 41-60: Neutral or sideways movement
   * 61-80: Good progress toward goal
   * 81-100: Optimal progress, aligns with all the provided strategies and hypotheses

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