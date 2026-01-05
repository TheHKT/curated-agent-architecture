from tinydb import Query
from utils.util import dbToString
import json
import uuid


class HypothesesRefiner:
    def __init__(self, client, model, hypothesesDb):
        self.client = client
        self.model = model
        self.hypothesesDb = hypothesesDb
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
        self.hypothesesDb.insert(newEntry)

    def modify(self, bullet_id, content):
        self.hypothesesDb.update({"content": content}, self.query.id == bullet_id)

    def remove(self, bullet_id):
        self.hypothesesDb.remove(self.query.id == bullet_id)

    def getTools(self):
        return [
            {
                "type": "function",
                "function": {
                    "name": "ADD",
                    "description": "Adds a new entry into the hypotheses to help future navigation tasks.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "section": {
                                "type": "string",
                                "description": "The section of the hypotheses to which the new entry should be added. If there is no appropriate section, it will create a new section with a relevant title. If you can please reuse existing sections.",
                            },
                            "content": {
                                "type": "string",
                                "description": "The content of the new entry to be added to the hypotheses. This could be a theory, an observation about the environment",
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
                    "description": "Removes an existing entry from the hypotheses that was falsified.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "bullet_id": {
                                "type": "string",
                                "description": "The identifier of the bulletpoint to be removed from the hypotheses.",
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
                    "description": "Modifies an existing entry in the hypotheses to improve its clarity, accuracy, or relevance.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "bullet_id": {
                                "type": "string",
                                "description": "The identifier of the bulletpoint to be modified.",
                            },
                            "content": {
                                "type": "string",
                                "description": "The content which will overwrite the existing entry.",
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
                     You are an LLM scientist analyzing a dynamic/non-deterministic 2D environment based on navigation trajectories. Your goal is to develop and refine hypotheses about how the environment works—NOT to describe specific layouts or navigation strategies.

                    # Your Task
                    Observe the trajectory and update your hypotheses about environmental dynamics, mechanics, and object behaviors. Focus on:
                    - What different symbols/objects represent
                    - How objects interact or transform
                    - Rules governing environmental dynamics
                    - Cause-and-effect relationships

                    # Guidelines
                    - DO: Describe general mechanics, object properties, and interaction rules
                    - DO NOT: Describe specific layouts, positions, or navigation strategies
                    - Keep hypotheses concise and testable
                    - If observations contradict existing hypotheses, use REMOVE or MODIFY
                    - If you discover new mechanics, use ADD

                    # Input Context
                    CURRENT_HYPOTHESES_START
                    {dbToString(self.hypothesesDb)}
                    CURRENT_HYPOTHESES_END

                    TRAJECTORY_START
                    {trajectory}
                    TRAJECTORY_END

                    # Instructions
                    Based on this trajectory, what hypotheses about environmental dynamics should be added, modified, or removed? Use only the provided tools to update your hypotheses.
                    Keep the hypotheses concise and try to merge similar ideas into single entries where possible.
                     """,
            }
        ]
        return initial_prompt