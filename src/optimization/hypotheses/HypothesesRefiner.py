from tinydb import Query
from utils.util import call_llm, dbToString, execute_tool_call
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

        msg = call_llm(self.client, self.model, prompt, self.getTools(), debug)

        if msg and msg.tool_calls:
            if debug:
                print(f"== Num_Tools: {len( msg.tool_calls)}")
            for tool_call in msg.tool_calls:
                tool_response = execute_tool_call(self.TOOL_MAPPING, tool_call, debug)

        return msg.content

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
                    "description": "Adds a new entry into the hypotheses playbook to help future navigation tasks. Keep ADD atomic and only add on bulletpoint per operation!",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "section": {
                                "type": "string",
                                "description": "The section of the hypotheses to which the new entry should be added. If there is no appropriate section, it will create a new section with a relevant title. If you can please reuse existing sections.",
                            },
                            "content": {
                                "type": "string",
                                "description": "The content of the new entry to be added to the hypotheses. This could be a theory, an observation about the environment.",
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
                    "description": "Removes an existing entry from the hypotheses playbook that was falsified.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "bullet_id": {
                                "type": "string",
                                "description": "The identifier of the bulletpoint to be removed from the hypotheses playbook.",
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
                "content": 
f"""
You are an LLM scientist analyzing a dynamic/non-deterministic 2D environment based on past navigation trajectories.
Your goal is to develop and refine a hypotheses playbook about how the environment works—NOT to describe specific layouts or navigation strategies.

# Inputs
## Current Hypotheses about the Environment:
{dbToString(self.hypothesesDb) if self.hypothesesDb is not None and dbToString(self.hypothesesDb) else 'No hypotheses available yet!'}

## Trajectory of the a navigation agent that traversed the environment:
{trajectory}

# Updating Process:
1. Observe the trajectory about environmental dynamics, mechanics, and object behaviors. Focus on:
    - What different symbols/objects represent
    - How objects interact or transform
    - Rules governing environmental dynamics
    - Cause-and-effect relationships
2. Analyze the current hypotheses in the light of these observations. Do they still hold? Are they falsified? Do they need modification for clarity or accuracy?
3. Use the provided tools to update your hypotheses; ensure they are concise, testable, and focused on mechanics and dynamics, not specific layouts or strategies.
4. Read through the updated hypotheses to ensure they are clear, non-redundant, and provide valuable insights about the environment.

# Guidelines:
- DO: Describe general mechanics, object properties, and interaction rules
- DO NOT: Describe specific layouts, positions, or navigation strategies
- If observations contradict existing hypotheses, use REMOVE or MODIFY
- If you discover new mechanics, use ADD

# Output
Your output should ONLY consist of tool calls to update the hypotheses playbook based on your analysis.
Do NOT include any additional text or explanations outside of the tool calls.
You MUST use the provided tools to update hypotheses. A description of how to use the tools is provided in the next section.
                    """,
            }
        ]
        return initial_prompt