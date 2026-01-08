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
        trajectory = self.prompts.getGeneratorPrompt(self.actualEnvActions.getState())
        state = StepState()

        while not state.isTerminated:
            state.wasSimulatedLastStep = state.isSimulated
            response = self.client.chat.completions.create(model=self.model, messages=trajectory, tools=self.tools).choices[0].message

            if debug: print(f"== Step {state.stepCounter} == Response: {response.content}")

            entry = {"role": response.role, "content": response.content}
            trajectory.append(entry)

            if response.tool_calls:
                try:
                    if(debug): print(f"== Tools called: {len(response.tool_calls)}")

                    for tool_call in response.tool_calls:
                        state.isSimulated = tool_call.function.name.startswith("simulate_")
                        
                        if(state.isSimulated):
                            self.executeSimulatedToolCall(tool_call, state, trajectory, response, debug)
                        else:
                            self.executeActualToolCall(tool_call, state, trajectory, response, debug)
                        
                except Exception as e:
                    if(debug): print(f"== Error occurred during tool execution! {str(e)}")
                    
                    entry = { "role": "system", "content": f"An error occurred during tool execution {"in the simulated environment" if state.isSimulated else ""}: {str(e)}"}
                    trajectory.append(entry)
                    
                    if state.isSimulated:
                        entry = {"role": response.role, "content": f"## Stopped the simulation ##\nI am now executing a tool in the actual environment!"}
                        trajectory.append(entry)
                    else: # only stop if the actual environment tool failed
                        sanitizedTrajectory = self.removeSimulatedSteps(trajectory)
                        return sanitizedTrajectory[1:], trajectory[1:], state.stepCounter, state.simulatedStepCounter #removed first init entry
            
            if state.isTerminated:
                sanitizedTrajectory = self.removeSimulatedSteps(trajectory)
                return sanitizedTrajectory[1:], trajectory[1:], state.stepCounter, state.simulatedStepCounter
            if state.isSimulated:
                state.simulatedStepCounter+=1 
            else:
                state.stepCounter+=1
         
    def executeSimulatedToolCall(self, tool_call, state, trajectory, response, debug):
        if(not state.wasSimulatedLastStep):
            entry = {"role": response.role, "content": f"## Starting Simulation ##\nI am now executing tools in the simulated environment!"}
            trajectory.append(entry)
        
        entry = self.executeToolCall(tool_call, self.simulatedEnvActions, state, trajectory, debug)
        trajectory.append(entry)        
                
    def executeActualToolCall(self, tool_call, state, trajectory, response, debug):
        if(state.wasSimulatedLastStep):
            entry = {"role": response.role, "content": f"## Stopped the simulation ##\nI am now executing a tool in the actual environment!"}
            trajectory.append(entry)
        
        entry = self.executeToolCall(tool_call, self.actualEnvActions, state, trajectory, debug)
        trajectory.append(entry) # TODO: [1]
        self.simulatedEnvActions.refreshSimulatedEnv(self.actualEnvActions)  
    
    def executeToolCall(self, tool_call, envActions, state, trajectory, debug):
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
        trajectory.append(entry)
        # update trajectory ifSimulation was terminated
        if tool_response["isTerminated"] and state.isSimulated:
            entry = {"role": "system", "content": f"## Stopped the simulation ##\nThe simulation was terminated by the environment."}
            trajectory.append(entry)
            self.simulatedEnvActions.refreshSimulatedEnv(self.actualEnvActions)
        if(debug): self.printToolCall(tool_call, tool_response)
        return entry 
            
    def printToolCall(self, tool_call, tool_response):
        print(f"== Tool: {tool_call.function.name}")
        print(f"== Tool Parameters: {tool_call.function.arguments}")  
        print(f'== New State:\n {tool_response["state"]}')   
        print("======")
        
    def removeSimulatedSteps(self, trajectory):
        sanitizedTrajectory = []
        i = 0
        while i < len(trajectory):
            entry = trajectory[i]
            content = entry.get("content", "")
            if content.startswith("## Starting Simulation ##"):
                sanitizedTrajectory.append({"role": "system", "content": "Executed a simulation"})
                
                while i < len(trajectory) and not trajectory[i].get("content", "").startswith("## Stopped the simulation ##"):
                    i += 1
                if i < len(trajectory):
                    i += 1
            else:
                sanitizedTrajectory.append(entry)
                i += 1
        return sanitizedTrajectory