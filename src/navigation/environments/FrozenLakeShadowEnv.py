from utils.util import dbToString
import json
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

                            CRITICAL: Return ONLY the JSON object. No explanations, no markdown, no additional text.
                            """,
        }
    ]
        response = (
            self.client.chat.completions.create(
                model=self.model, messages=prompt
            )
            .choices[0]
            .message
        ).content

        if debug:
            print("Best Move:", response)
        
        return self.extract_move_from_response(response, moves)

    def eval_best_sample(self, samples: list[tuple[str, int]]) -> list[tuple[str, int]]:
        """
        This method evaluates which of the generated sample trajectories is the best performing and returns it.
        
        :param samples: A list of sample trajectories to evaluate.
        :return: The best performing trajectory.
        :rtype: List[str]
        """
        # TODO: this is a placeholder
        return random.choice(samples)
    
    def extract_move_from_response(self, response: str, moves: list[str]) -> tuple[str, int]:
        """
        This method extracts the move from the LLM response. Sometimes the LLM returns extra text or malformed JSON, so this method cleans it up.
        
        :param response: The LLM response containing the move.
        :return: The extracted move and its value as a string int tuple.
        :rtype: tuple[str, int]:
        """
        try:            
            cleaned_response = response.strip()
            if cleaned_response.startswith('.'):
                cleaned_response = cleaned_response[1:].strip()

            start_idx = cleaned_response.find('{')
            end_idx = cleaned_response.rfind('}')

            if start_idx != -1 and end_idx != -1:
                json_str = cleaned_response[start_idx:end_idx+1]
                response_dict = json.loads(json_str)

                if response_dict["move"] in moves:
                    return (response_dict["move"], response_dict["value"])
                else:
                    print(f"Warning: LLM returned invalid move '{response_dict['move']}', using random.")
                    return (random.choice(moves), 50)
            else:
                raise json.JSONDecodeError("No JSON object found", cleaned_response, 0)

        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error parsing response: {e}")
            print(f"Raw response: {response}")
            return (random.choice(moves), 50)