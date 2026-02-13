import uuid 
from tinydb import Query
from utils.util import dbToString, execute_tool_call, call_llm

class PlaybookRefactorer:
    def __init__(self, client, model, playbook):
        self.client = client
        self.model = model
        self.playbook = playbook
        self.query = Query()
        
        self.TOOL_MAPPING = {
          "ADD": self.add,
          "REMOVE": self.remove,
          "MODIFY": self.modify
      }

    def run(self, debug=False):         
        msg = call_llm(self.client, self.model, self.getPrompt(), self.getTools(), debug)
            
        if msg and msg.tool_calls:
            if(debug):
                print(f"== Num_Tools: {len( msg.tool_calls)}")          
            for tool_call in msg.tool_calls:
                tool_response = execute_tool_call(self.TOOL_MAPPING, tool_call, debug)

        return msg.content
    
    def add(self, section, content):
        newEntry = {"id": str(uuid.uuid4()), "section": section, "content": content}
        self.playbook.insert(newEntry)
    def modify(self, bullet_id, content):
        self.playbook.update({"content": content}, self.query.id == bullet_id)
    def remove(self, bullet_id):
        self.playbook.remove(self.query.id == bullet_id)

    def getPrompt(self):
        return [
            {
                "role": "system",
                "content": f'''
You are a playbook refactorer.
Your job is to consolidate and optimize a given playbook by removing duplicates, resolving contradictions, and improving the clarity and usefulness of the entries.
The playbook consists of various sections with bullet points that provide guidance.

# Inputs
## Current Playbook             
{dbToString(self.playbook) if self.playbook is not None and dbToString(self.playbook) else 'No playbook available yet!'}

# Your Refactoring Task:
1. Read each entry in the playbook and understand them
2. Go through each entry again and find redundancies, contradictions, or areas for improvement in the playbook entries:
 - If you find redundant entries: Merge them into a single entry. Merging means that you should remove one of the redundant entries and modify the other.
 - If you find contradictions, decide which entry is more helpful and remove or modify the less helpful one.
 - If you find entries that can be improved, modify them to be clearer and more actionable.

## Critical: 
Do NOT try to generate new insights or strategies that are not already present in the playbook.
Your task is to optimize and refactor the existing content, not to create new content.
It is very important that there are no duplicates or contradictions in the playbook, and that all entries are as clear and helpful as possible.
            
# Output
Your output should ONLY consist of tool calls to update the playbook based on your decisions.
Do NOT include any additional text or explanations outside of the tool calls.
You MUST use the provided tools to update the playbook.
A description of how to use the tools is provided in the next section.
Always ensure that each tool call is in a valid JSON format.
'''
            }
        ]

    def getTools(self):
        return [
    {
        "type": "function",
        "function": {
            "name": "ADD",
            "description": "Adds a new entry to the playbook.",
            "parameters": {
                "type": "object",
                "properties": {
                    "section": {
                        "type": "string",
                        "description": "Section name (e.g., 'General Navigation Heuristics'). Reuse existing sections when possible."
                    },
                    "content": {
                        "type": "string",
                        "description": "Single bulletpoint for guidance."
                    }
                },
                "required": ["section", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "REMOVE",
            "description": "Remove an entry from the playbook.",
            "parameters": {
                "type": "object",
                "properties": {
                    "bullet_id": {
                        "type": "string",
                        "description": "ID of the entry to remove."
                    }
                },
                "required": ["bullet_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "MODIFY",
            "description": "Merge or improve an existing entry.",
            "parameters": {
                "type": "object",
                "properties": {
                    "bullet_id": {
                        "type": "string",
                        "description": "ID of the entry to modify."
                    },
                    "content": {
                        "type": "string",
                        "description": "New content that completely replaces the existing entry."
                    }
                },
                "required": ["bullet_id", "content"]
            }
        }
    }
]