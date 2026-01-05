class Navigator():
    def __init__ (self, environment, shadow_env):
        """        
        :param environment: The actual environment in which the navigation takes place.
        :param shadow_env: The shadow environment used for lookahead simulations to get the best possible next_move.
        """

        self.environment = environment
        self.shadow_env = shadow_env

    def run(self, debug=False):
        trajectory = "# Navigation Trajectory #\n\n"
        step_counter = 1
        is_terminated = False
        self.environment.reset()

        while not is_terminated:
            trajectory += f"## Step {step_counter} ##\n"
            
            state = self.environment.get_state() # maybe return one string reprsenting that state, and the actual env state
            trajectory += f"Current State:\n{state}\n\n"
            
            next_move = self.shadow_env.get_next_move(self.environment)
            trajectory += f"Execute Move: {next_move}\n"
            
            response = self.environment.ACTION_MAP[next_move]()
            trajectory += f"New State:\n{response['state']}\n\n"
            
            is_terminated = response["is_terminated"]
            trajectory += f"{'Navigation stopped' if is_terminated else 'Navigation continues'}\n"

            step_counter += 1
            trajectory += "##############################\n\n"

            if debug:
                print(f"## Step {step_counter - 1} ##\n")
                print(f"Current State:\n{state}\n\n")
                print(f"Execute Move: {next_move}\n")
                print(f"New State:\n{response['state']}\n\n")
                print(f"{'Navigation stopped' if is_terminated else 'Navigation continues'}\n")
                print("##############################\n\n")

        return trajectory