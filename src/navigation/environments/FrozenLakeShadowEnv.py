from utils.util import dbToString, extract_json_from_llm_response
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
            "content": f"""You are an expert value estimator for a reinforcement learning agent operating in a dynamic 2D environment.

                            # Task
                            Analyze the current state, hypotheses, and proven strategies to select the optimal move and estimate its value.

                            # Context
                            ## Environment Hypotheses
                            These are current beliefs about how the environment operates:
                            {dbToString(self.hypothesesDb)}

                            ## Proven Strategies
                            These strategies have been successful in similar situations:
                            {dbToString(self.strategiesDb)}

                            ## Current State
                            {state}

                            ## Available Moves
                            {', '.join(moves)}

                            # Analysis Process
                            1. Review each hypothesis and identify which apply to the current state
                            2. Consider which proven strategies are relevant given the hypotheses
                            3. For each available move, evaluate alignment with strategies and expected outcomes
                            4. Select the move with highest expected value

                            # Output Requirements
                            Return ONLY a valid JSON object with this exact structure:
                            {{"move": "move_name", "value": 75}}

                            - move: Must be exactly one move from the Available Moves list
                            - value: Integer from 0-100 where:
                              * 0-20: Poor outcome, contradicts strategies
                              * 21-40: Below average, minimal benefit
                              * 41-60: Moderate outcome, some strategic alignment
                              * 61-80: Good outcome, strong strategic fit
                              * 81-100: Optimal outcome, maximizes objectives

                            CRITICAL: Return ONLY the JSON object. No explanations, no markdown, no additional text. Failure to comply will result in invalid output and errors.
                            """,
        }
    ]
        response = (
            self.client.chat.completions.create(
                model=self.model, messages=prompt
            )
            .choices[0]
            .message
        )

        response_dict = extract_json_from_llm_response(response.content)

        if response_dict is not None and "move" in response_dict and "value" in response_dict and response_dict["move"] in moves:
            return (response_dict["move"], response_dict["value"])
        else:
            print(f"Warning: LLM returned invalid response '{response_dict}', using random.")
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