from tinydb import TinyDB, Query
import json
import uuid


class EnvironmentCurator:
    def __init__(self, envDescriptionPath):
        envDescriptionPath = "../../store/frozenlake/env_descriptions.json"  # TODO: hardcoded for debugging

        self.envDescriptionDB = TinyDB(envDescriptionPath)
        self.query = Query()
        self.TOOL_MAPPING = {
            "ADD": self.add,
            "REMOVE": self.remove,
            "MODIFY": self.modify,
        }

    def run(self, trajectory, debug=False):
        prompt = self.generateInitialPrompt(trajectory)

        response = (
            self.client.chat.completions.create(
                model=self.model, messages=prompt, tools=self.getTools()
            )
            .choices[0]
            .message
        )

        if debug:
            print(f"== Response: {response.content}")

        if response.tool_calls:

            if debug:
                print(f"== Num_Tools: {len( response.tool_calls)}")

            for tool_call in response.tool_calls:
                tool_name = tool_call.function.name
                tool_response = (
                    self.TOOL_MAPPING[tool_name](
                        **(json.loads(tool_call.function.arguments))
                    )
                    if tool_call.function.arguments is not None
                    else self.TOOL_MAPPING[tool_name]()
                )

                if debug:
                    print(f"== Tool: {tool_name}")
                    print(f"== Tool Parameters: {tool_call.function.arguments}")

        return response.content

    def add(self, section, content):
        newEntry = {"id": str(uuid.uuid4()), "section": section, "content": content}
        self.envDescriptionDB.insert(newEntry)

    def modify(self, bullet_id, content):
        self.envDescriptionDB.update({"content": content}, self.query.id == bullet_id)

    def remove(self, bullet_id):
        self.envDescriptionDB.remove(self.query.id == bullet_id)

    def getTools(self):
        return [
            {
                "type": "function",
                "function": {
                    "name": "ADD",
                    "description": "Adds a new entry into the environment description to help future navigation tasks.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "section": {
                                "type": "string",
                                "description": "The section of the environment description to which the new entry should be added. If there is no appropriate section, it will create a new section with a relevant title. If you can please reuse existing sections.",
                            },
                            "content": {
                                "type": "string",
                                "description": "The content of the new entry to be added to the environment description. This could be a theory, an observation about the environment",
                            },
                        },
                        "required": ["section", "content"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "REMOVE",
                    "description": "Removes an existing entry from the environment description that was falsified.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "bullet_id": {
                                "type": "string",
                                "description": "The identifier of the bulletpoint to be removed from the environment description.",
                            }
                        },
                        "required": ["bullet_id"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "MODIFY",
                    "description": "Modifies an existing entry in the environment description to improve its clarity, accuracy, or relevance.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "bullet_id": {
                                "type": "string",
                                "description": "The identifier of the bulletpoint to be modified to the environment description.",
                            },
                            "content": {
                                "type": "string",
                                "description": "The content which will overwrite the existing entry in the environment description.",
                            },
                        },
                        "required": ["bullet_id", "content"],
                    },
                },
            },
        ]

    def generateInitialPrompt(self, trajectory):
        initial_prompt = [
            {
                "role": "system",
                "content": f"""
                     You are a LLM scienties trying to derive how a dynamic 2D environment works based on a given navigation trajectory. Your goal is to update and curate your environment description based on the observations from the trajectory.
                     Please note that you are ONLY allowed to describe how the environments works. Please do NOT include any strategies for navigation or movement in the environment description. Remember you are a scientiest that tries to observe and describe the environment, you are dont navigate it.

                    To modify the given environment description, you can ONLY use the provided tools, that are describe fruther below.
                    Please try to keep your environment description as concise as possible, while still being accurate and complete.
                    If you see something in the trajectory that contradicts your current environment description, please use the REMOVE or MODIFY tool to update your environment description accordingly.
                    If you see something new in the trajectory that is not yet described in your environment description, please use the ADD tool to add a new entry to your environment description.

                     ## Input Context
                     ENVIRONMENT_DESCRIPTION_START
                     {self.envDescriptionToString()}
                     ENVIRONMENT_DESCRIPTION_END

                     TRAJECTORY_START
                     {trajectory}
                     TRAJECTORY_END
                     """,
            }
        ]
        return initial_prompt

    def envDescriptionToString(self):
        allEntries = self.envDescriptionDB.all()
        
        sections_dict = {}
        for entry in allEntries:
            section = entry['section']
            if section not in sections_dict:
                sections_dict[section] = []
            sections_dict[section].append(entry)
        
        output = []
        for section, entries in sections_dict.items():
            output.append(f"## {section}")
            output.append("")
            for entry in entries:
                output.append(f"- {entry['id']}: {entry['content']}")
                output.append("")
        
        return "\n".join(output)