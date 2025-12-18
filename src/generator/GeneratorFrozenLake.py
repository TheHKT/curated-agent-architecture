from Generator import Generator


class GeneratorFrozenLake(Generator):
    def generateInitialPrompt(self):
        initial_prompt = [
            {
                "role": "system",
                "content": f"""
                     You are a multi-turn LLM Agent Navigator in a dynamic 2D environment. Your goal: reach position G from current position [] using the provided tools for movement.

                     ## Multi-Turn Operation
                     You receive a NAVIGATION TRAJECTORY containing:
                     - All your previous decisions and reasoning
                     - Environment responses after each action
                     - Current state resulting from your last move

                     Your output becomes input for your next iteration. Each decision builds on this growing trace, so reason clearly to help your future self.

                     ## Core Task
                     Read the playbook and the reflection -> Apply rules, knowledge and strategies retrieved from those documents -> Decide the next one move from the given context

                     ## Decision Process
                     1. **Apply Learning**
                        - PLAYBOOK: Proven strategies for similar situations, common failure patterns, environment dynamics
                        - REFLECTION: What worked and what failed, corrected approaches

                     2. **Construct your next move** (from trajectory)
                        - Use the learned strategies to reason

                     3. **Execute Single Move**
                        - Reason concisely (this persists in your trace)
                        - Call exactly ONE tool: move_left, move_right, move_up, move_down
                        - Environment responds → new state → next iteration begins

                     ## Critical Constraints
                     ✓ Remember you are navigating a dynamic 2D environment where the reaction of the environment to your moves is non-deterministic.
                     ✓ One tool call per turn (you're multi-turn, not multi-action)
                     ✓ State only updates after environment processes your move
                     ✓ Your reasoning propagates forward—be clear and useful
                     ✗ Never call multiple movement tools
                     ✗ Don't assume behaviors not observed in the trajectory

                     ## Input Context
                     PLAYBOOK_START
                     {self.policy if self.policy is not None else 'No playbook available'}
                     PLAYBOOK_END

                     ENVIRONMENT_DESCRIPTION_START
                     {self.envDescription}
                     ENVIRONMENT_DESCRIPTION_END

                     STATE_START
                     {self.actualEnv.getStateDescription() if self.actualEnv is not None else 'State unavailable'}
                     STATE_END
                     """,
            }
        ]
        return initial_prompt

    def getTools(self):
        return [
            {
                "type": "function",
                "function": {
                    "name": "move_left",
                    "description": "Moves the player left in the 2D environment. It returns the new state after the move was executed, the reward obtained and whether the task is terminated.",
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "move_right",
                    "description": "Moves the player right in the 2D environment. It returns the new state after the move was executed, the reward obtained and whether the task is terminated.",
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "move_up",
                    "description": "Moves the player up in the 2D environment. It returns the new state after the move was executed, the reward obtained and whether the task is terminated.",
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "move_down",
                    "description": "Moves the player down in the 2D environment. It returns the new state after the move was executed, the reward obtained and whether the task is terminated.",
                },
            },
        ]
