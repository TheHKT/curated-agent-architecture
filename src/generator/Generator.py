import json

from abc import ABC, abstractmethod

class Generator(ABC):
    def __init__ (self, client, model, actualEnv, shadowEnv):
        self.client = client
        self.model = model

        # Environments for the generator
        self.actualEnv = actualEnv
        self.shadowEnv = shadowEnv

        # Prompts for the generator
        self.tools = self.actualEnv.getGeneratorTools()

    def run(self, debug=False):
        prompt = self.actualEnv.getInitialGeneratorPrompt()
        trajectory = [] # The thoughts process, the tool calls and the states without the setup prompts

        counter = 1
        isTeriminated = False

        while not isTeriminated:
            response = self.client.chat.completions.create(model=self.model, messages=prompt, tools=self.tools).choices[0].message

            if debug:
                print(f"== Step {counter} ==")
                print(f"== Response: {response.content}")

            entry = {"role": response.role, "content": response.content}
            prompt.append(entry)
            trajectory.append(entry)

            ### TODO: Check wether the model wants to call a tool in the shadow environment or the actual environment
            if response.tool_calls:
                try:
                    if(debug):
                            print(f"== Tools called: {len( response.tool_calls)}")

                    for tool_call in response.tool_calls:
                        tool_name = tool_call.function.name
                        tool_response = self.actualEnv.ACTION_MAP[tool_name](**(json.loads(tool_call.function.arguments))) if tool_call.function.arguments is not None else self.actualEnv.ACTION_MAP[tool_name]()
                        isTerminated = tool_response["isTerminated"]
                        entry = {
                          "role": "tool",
                          "tool_call_id": tool_call.id,
                          "tool_name": tool_name,
                          "tool_arguments": tool_call.function.arguments,
                          "content": json.dumps(tool_response),
                        }
                        prompt.append(entry)
                        trajectory.append(entry)

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
                    trajectory.append(entry)
                    if(debug):
                        print(f"== Error occurred during tool execution! Terminating this iteration!")
                        print(f"== {str(e)}")

                    return trajectory, counter

            if isTerminated:
                return trajectory, counter
            ### END OF REFACTOR BLOCK

            counter+=1