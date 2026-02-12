from navigation.environments.ShadowEnvironment import ShadowEnvironment
from navigation.environments.Environment import Environment

class Navigator():
    def __init__ (self, environment: Environment, shadow_env: ShadowEnvironment):
        """        
        :param environment: The actual environment in which the navigation takes place.
        :param shadow_env: The shadow environment used for lookahead simulations to get the best possible next_move.
        """

        self.environment = environment
        self.shadow_env = shadow_env

    def run(self, sample_size=3, depth=4, max_nav_steps=30, use_llm_action=False, debug=False) -> str:
        trajectory = ""
        step_counter = 1
        is_terminated = False
        responses = []
        self.environment.reset()

        while not is_terminated:
            
            if step_counter > max_nav_steps:
                trajectory_entry = f"Reached maximum navigation steps ({max_nav_steps}).\n"
                trajectory_entry += "Navigation was terminated.\n"
                trajectory_entry += f"Final State:\n{responses[-1]['state']}\n"
                trajectory += trajectory_entry + "\n"
                if debug:
                    print("Reached maximum navigation steps; terminating.")
                break
            
            trajectory_entry = f"### Step {step_counter}:\n"
            
            state = self.environment.get_state() # maybe return one string reprsenting that state, and the actual env state
            trajectory_entry += f"Current State:\n{state}\n\n"
            
            # TODO: added the rated moves to the trace, such that refiner+curator can optimize the policies
            best_path = self.shadow_env.get_next_move(self.environment, sample_size=sample_size, depth=depth, use_llm_actions=use_llm_action, debug=debug)
            trajectory_entry += f"Performing lookahead with depth={depth} and sample_size={sample_size}...\n"
            trajectory_entry += self.path_to_string(best_path)

            next_move = best_path[0][0]
            trajectory_entry += f"Executed Move: {next_move}\n"
            response = self.environment.ACTION_MAP[next_move]()
            responses.append(response)
            trajectory_entry += f"Reward Received: {response['reward']}\n"

            is_terminated = response["is_terminated"]

            if is_terminated:
                trajectory_entry += "Navigation was terminated.\n"
                trajectory_entry += f"Final State:\n{response['state']}"

            trajectory += trajectory_entry + "\n"
            step_counter += 1

            if debug:
                print(trajectory_entry)

        return trajectory, responses
    
    def path_to_string(self, path):
        str = "Found best path:\n"
        for step, (move, value) in enumerate(path):
            str += f"- Step {step + 1}: Move: {move}, Value: {value}\n"
        str += "\n"
        return str