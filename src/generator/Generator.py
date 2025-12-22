import json
from abc import ABC

class Generator(ABC):
    def __init__ (self, client, model, actualEnvActions, simulatedEnvActions, prompts):
        self.client = client
        self.model = model

        # Actions in the environments for the generator
        self.actualEnvActions = actualEnvActions
        self.simulatedEnvActions = simulatedEnvActions
        self.prompts = prompts

        # Prompts for the generator
        self.tools = self.prompts.getGeneratorTools()

    def run(self, debug=False):
        prompt = self.prompts.getGeneratorPrompt(self.actualEnvActions.getState())
        trajectory = [] # The thoughts process, the tool calls and the states without the setup prompts

        stepCounter = 1
        simulatedStepCounter = 1
        isTeriminated = False

        while not isTeriminated:
            response = self.client.chat.completions.create(model=self.model, messages=prompt, tools=self.tools).choices[0].message

            if debug:
                print(f"== Step {stepCounter} ==")
                print(f"== Response: {response.content}")

            entry = {"role": response.role, "content": response.content}
            prompt.append(entry)
            trajectory.append(entry) # TODO: [1] monitor wether the simulated tool calls should be part of the trajectory or not

            if response.tool_calls:
                try:
                    if(debug):
                            print(f"== Tools called: {len( response.tool_calls)}")

                    for tool_call in response.tool_calls:
                        tool_name = tool_call.function.name
                        isSimulated = tool_name.startswith("simulate_")
                        
                        if(isSimulated):
                            tool_response = self.simulatedEnvActions.ACTION_MAP[tool_name](**(json.loads(tool_call.function.arguments))) if tool_call.function.arguments is not None else self.simulatedEnvActions.ACTION_MAP[tool_name]()
                        else:
                            tool_response = self.actualEnvActions.ACTION_MAP[tool_name](**(json.loads(tool_call.function.arguments))) if tool_call.function.arguments is not None else self.actualEnvActions.ACTION_MAP[tool_name]()
                        
                        isTerminated = tool_response["isTerminated"] and not isSimulated
                        entry = {
                          "role": "tool",
                          "tool_call_id": tool_call.id,
                          "tool_name": tool_name,
                          "tool_arguments": tool_call.function.arguments,
                          "content": json.dumps(tool_response),
                          "is_simulated": isSimulated
                        }
                        prompt.append(entry)
                        if not isSimulated: 
                            self.simulatedEnvActions.refreshSimulatedEnv(self.actualEnvActions) # if it was not simulated, the actual env has changed, therefore we must update the simulated env
                            trajectory.append(entry) # TODO: [1] 

                        if(debug):
                            print(f"== Tool: {tool_name}")
                            print(f"== Tool Parameters: {tool_call.function.arguments}")  
                            print(f'== New State:\n {tool_response["state"]}')   
                            print("======")  
                except Exception as e:
                    entry = {
                        "role": "system",
                        "content": f"An error occurred during tool execution: {str(e)}"
                    }
                    prompt.append(entry)
                    
                    if(debug):
                        print(f"== Error occurred during tool execution!")
                        print(f"== {str(e)}")

                    if not isSimulated: # only stop if the actual environment tool failed
                        trajectory.append(entry)
                        return trajectory, stepCounter, simulatedStepCounter

            if isTerminated:
                return trajectory, stepCounter, simulatedStepCounter

            if isSimulated:
                simulatedStepCounter+=1 
            else:
                stepCounter+=1