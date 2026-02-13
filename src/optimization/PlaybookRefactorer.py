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
You are a playbook refactorer. Your job is to consolidate and optimize playbooks by removing duplicates, resolving contradictions, and improving clarity.

# Input
## Current Playbook
{dbToString(self.playbook) if self.playbook is not None and dbToString(self.playbook) else 'No playbook available yet!'}

# Task
Refactor the playbook by:

1. **Identifying issues:**
   - Duplicate entries (same or highly similar guidance)
   - Contradictory entries (conflicting advice)
   - Unclear or vague entries

2. **Resolving duplicates (REQUIRED):**
   - Delete one duplicate entry entirely
   - Modify the remaining entry to incorporate any unique information from the deleted duplicate
   - Do NOT keep both duplicates under any circumstances

3. **Resolving contradictions:**
   - Determine which entry provides better guidance
   - Delete or modify the less helpful entry
   - If both have merit, merge them into a single coherent entry

4. **Improving clarity:**
   - Rewrite vague entries to be specific and actionable
   - Ensure consistent formatting and terminology
   - Remove unnecessary verbosity

# Critical Constraints
- Work ONLY with existing content—do not create new insights or strategies
- Eliminate ALL duplicates and contradictions
- Be aggressive about merging—prefer one clear entry over multiple similar ones
- Every modification must make the playbook more useful

# Output Format
- Use ONLY tool calls to update the playbook
- No explanatory text outside tool calls
- Each tool call must be valid JSON
- Complete all necessary updates in a single response

Remember: Your goal is a lean, contradiction-free playbook with zero redundancy.
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