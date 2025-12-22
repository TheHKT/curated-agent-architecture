from environments.FrozenLakeActions import FrozenLakeActions
from environments.FrozenLakeActions import Move
from utils.util import dbToString
import copy
import json

from typing import Dict, Any

class FrozenLakeEnvEmulator(FrozenLakeActions):
    def __init__(self, actualEnv, useLlm=False, hypothesesDb=None, client=None, model=None):
        self.useLlm = useLlm
        self.client = client
        self.model = model
        self.hypothesesDb = hypothesesDb
        self.llmState = actualEnv.getState()
        
        super().__init__(copy.deepcopy(actualEnv.env))
        
    def getState(self, player_pos: int | None = None) -> str:
        if self.useLlm:
            return self.llmState
        else:
            return super().getState(player_pos)
    
    def executeAction(self, action: Move) -> Dict[str, Any]:
        if self.useLlm:
            prompt = [
                {"role": "system", 
                 "content": f'''
                 You are an outstanding emulator for dynamic environments. Your job is to predict the next state of the environment based on the current state, the action taken and a set of hypotheses.
                 
                 # Inputs
                 ## Current State of the Environment
                    {self.llmState}
                    
                ## Hypotheses about the Environment
                    {dbToString(self.hypothesesDb)}
                    
                ## Action Taken
                    {action.name}
                    
                # Output
                You must output the next state of the environment in JSON format as follows:
                {{
                    "state": state,
                    "reward": reward,
                    "isTerminated": isTerminated
                }}
                
                # Critical Instructions
                - Always respond in valid JSON format.
                - Ensure the state representation matches the format of the input state, including newline characters (\\n).
                 '''},
            ]
            
            responseStr = (
            self.client.chat.completions.create(model=self.model, messages=prompt)
                .choices[0]
                .message)
            response = json.loads(responseStr.content)
            self.llmState = response["state"]
            return response
        else:
            return super().executeAction(action)
        