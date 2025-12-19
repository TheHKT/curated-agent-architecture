import gymnasium as gym

from typing import Dict, Callable, Any
from enum import Enum
from utils.util import dbToString
from environments.Environment import Environment

class Move(Enum):
    LEFT = 0
    DOWN = 1
    RIGHT = 2
    UP = 3
    
class EnvironmentFrozenLake(Environment):
    def __init__(self, env = gym.make("FrozenLake-v1", render_mode="ansi",  desc=None, map_name="4x4", is_slippery=True, success_rate=2.0/3.0, reward_schedule=(1, 0, 0)), policyDb = None, hyptothesisDb = None):
        super().__init__(env)
        self.policyDb = policyDb
        self.hyptothesisDb = hyptothesisDb
        
    @property
    def ACTION_MAP(self) -> Dict[str, Callable]:
        return {
            "move_left": self.moveLeft,
            "move_right": self.moveRight,
            "move_up": self.moveUp,
            "move_down": self.moveDown
        }

    def getState(self, player_pos: int | None = None) -> str:
        if player_pos is None:
            player_pos = self.env.unwrapped.s
        desc = self.env.unwrapped.desc
        nrow, ncol = desc.shape

        row = player_pos // ncol
        col = player_pos % ncol

        result = []
        for r in range(nrow):
            line = ""
            for c in range(ncol):
                cell = desc[r][c].decode('utf-8') if isinstance(desc[r][c], bytes) else desc[r][c]
                if r == row and c == col:
                    line += f"[{cell}]"
                else:
                    line += f" {cell} "
            result.append(line)

        return '\n'.join(result)
    
    def moveLeft(self):
        return self.executeAction(Move.LEFT)
    def moveRight(self):
        return self.executeAction(Move.RIGHT)
    def moveUp(self):
        return self.executeAction(Move.UP)
    def moveDown(self):
        return self.executeAction(Move.DOWN)
    def executeAction(self, action: Move) -> Dict[str, Any]:
        observation, reward, isTerminated, truncated, info = self.env.step(action.value)
        state = self.getState(player_pos=observation)
        answer = {
            "state": state,
            "reward": reward,
            "isTerminated": isTerminated
        }
        return answer
    
    def getInitialGeneratorPrompt(self) -> str:
        return [
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
                     {dbToString(self.policyDb) if self.policyDb is not None else 'No playbook available'}
                     PLAYBOOK_END

                     HYPOTHESIS_START
                     {dbToString(self.hyptothesisDb) if self.hyptothesisDb is not None else 'No environment description available'}
                     HYPOTHESIS_END

                     STATE_START
                     {self.getState()}
                     STATE_END
                     """,
            }
        ]
    def getGeneratorTools(self) -> str:
        return [
            {
                "type": "function",
                "function": {
                    "name": "move_left",
                    "description": "Moves the player left in the 2D environment. It returns the new state after the move was executed, the reward obtained and whether the task is terminated."
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "move_right",
                    "description": "Moves the player right in the 2D environment. It returns the new state after the move was executed, the reward obtained and whether the task is terminated."
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "move_up",
                    "description": "Moves the player up in the 2D environment. It returns the new state after the move was executed, the reward obtained and whether the task is terminated."
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "move_down",
                    "description": "Moves the player down in the 2D environment. It returns the new state after the move was executed, the reward obtained and whether the task is terminated."
                }
            }
        ]