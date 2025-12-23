import json
from abc import ABC
from dataclasses import dataclass

@dataclass
class StepState:
    stepCounter: int = 1
    simulatedStepCounter: int = 1
    isTerminated: bool = False
    isSimulated: bool = False
    wasSimulatedLastStep: bool = False  

class Generator(ABC):
    def __init__ (self, client, model, actualEnvActions, simulatedEnvActions, prompts, maxSteps=25, maxSimulatedSteps=10):
        self.client = client
        self.model = model

        # Actions in the environments for the generator
        self.actualEnvActions = actualEnvActions
        self.simulatedEnvActions = simulatedEnvActions
        self.prompts = prompts

        # Prompts for the generator
        self.tools = self.prompts.getGeneratorTools()

    def run(self, debug=False):
        fullTrajectory = self.prompts.getGeneratorPrompt(self.actualEnvActions.getState())
        compactTrajectory = [] # without the setup prompts and simulated tool calls
        state = StepState()

        while not state.isTerminated:
            response = self.client.chat.completions.create(model=self.model, messages=fullTrajectory, tools=self.tools).choices[0].message

            if debug: print(f"== Step {state.stepCounter} == Response: {response.content}")

            entry = {"role": response.role, "content": response.content}
            fullTrajectory.append(entry)
            compactTrajectory.append(entry)

            if response.tool_calls:
                try:
                    if(debug): print(f"== Tools called: {len(response.tool_calls)}")

                    for tool_call in response.tool_calls:
                        state.wasSimulatedLastStep = state.isSimulated
                        state.isSimulated = tool_call.function.name.startswith("simulate_")
                        
                        if(state.isSimulated):
                            self.executeSimulatedToolCall(tool_call, state, fullTrajectory, compactTrajectory, response, debug)
                        else:
                            self.executeActualToolCall(tool_call, state, fullTrajectory, compactTrajectory, response, debug)
                        
                except Exception as e:
                    entry = { "role": "system", "content": f"An error occurred during tool execution {"in the simulated environment" if state.isSimulated else ""}: {str(e)}"}
                    fullTrajectory.append(entry)
                    compactTrajectory.append(entry)
                    
                    if(debug): print(f"== Error occurred during tool execution! {str(e)}")
                    
                    if not state.isSimulated: # only stop if the actual environment tool failed
                        return compactTrajectory, state.stepCounter, state.simulatedStepCounter
            
            if state.isTerminated:
                return compactTrajectory, state.stepCounter, state.simulatedStepCounter
            if state.isSimulated:
                state.simulatedStepCounter+=1 
            else:
                state.stepCounter+=1
         
    def executeSimulatedToolCall(self, tool_call, state, prompt, trajectory, response, debug):
        entry = {"role": response.role, "content": f"Starting Simulation.I am now executing tools in the simulated environment!"}
        prompt.append(entry)
        trajectory.append(entry)
        
        self.executeToolCall(tool_call, self.simulatedEnvActions, state, prompt, trajectory, debug)        
                
    def executeActualToolCall(self, tool_call, state, prompt, trajectory, response, debug):
        if(state.wasSimulatedLastStep):
            entry = {"role": response.role, "content": f"Stopped the simulation. I am now executing a tool in the actual environment!"}
            prompt.append(entry)
            trajectory.append(entry)
        
        entry = self.executeToolCall(tool_call, self.actualEnvActions, state, prompt, trajectory, debug)
        trajectory.append(entry) # TODO: [1]
        self.simulatedEnvActions.refreshSimulatedEnv(self.actualEnvActions)  
    
    def executeToolCall(self, tool_call, envActions, state, prompt, trajectory, debug):
        tool_response = envActions.ACTION_MAP[tool_call.function.name](**(json.loads(tool_call.function.arguments))) if tool_call.function.arguments is not None else envActions.ACTION_MAP[tool_call.function.name]()
        
        state.isTerminated = tool_response["isTerminated"] and not state.isSimulated
        entry = {
          "role": "tool",
          "tool_call_id": tool_call.id,
          "tool_name": tool_call.function.name,
          "tool_arguments": tool_call.function.arguments,
          "content": json.dumps(tool_response),
          "is_simulated": state.isSimulated
        }
        prompt.append(entry)
        if(debug): self.printToolCall(tool_call, tool_response)
        return entry 
            
    def printToolCall(self, tool_call, tool_response):
        print(f"== Tool: {tool_call.function.name}")
        print(f"== Tool Parameters: {tool_call.function.arguments}")  
        print(f'== New State:\n {tool_response["state"]}')   
        print("======")