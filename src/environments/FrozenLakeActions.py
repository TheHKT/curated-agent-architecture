import gymnasium as gym

from environments.Actions import Actions
from enum import Enum
from typing import Dict, Callable, Any

class Move(Enum):
    LEFT = 0
    DOWN = 1
    RIGHT = 2
    UP = 3

# This class implements the Actions interface for the FrozenLake environment.
class FrozenLakeActions(Actions):
    def __init__(self, env = gym.make("FrozenLake-v1", render_mode="ansi",  desc=None, map_name="4x4", is_slippery=True, success_rate=2.0/3.0, reward_schedule=(1, 0, 0))):
        super().__init__(env)
        
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