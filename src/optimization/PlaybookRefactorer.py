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
You are a playbook refactorer. Your job is to consolidate and optimize playbooks
by removing duplicates, flagging contradictions, and improving clarity where needed.

# Input
## Current Playbook
{dbToString(self.playbook) if self.playbook is not None and dbToString(self.playbook) else 'No playbook available yet!'}

# Core Definitions
**Duplicate**: Two entries that convey the same information, regardless of wording
or section label. Semantic equivalence matters more than textual similarity —
rephrasings and cross-section duplicates must be caught.

**Contradiction**: Two entries that give conflicting guidance or assert
incompatible facts (both cannot be true at the same time).

# Task
## Step 1 — Audit every entry pair
Compare ALL pairs of entries across ALL sections. Do not limit comparison to
entries within the same section. Classify each pair as: duplicate, contradiction,
or independent.

## Step 2 — Resolve duplicates
When two entries are duplicates, issue exactly 2–3 tool calls:
1. REMOVE tool call → delete the weaker entry
2. UPDATE tool call (only if needed) → update the surviving entry to incorporate
   any unique information or wording from the deleted entry
3. (No further calls — never keep both)

Decision rule for which entry to delete, applied IN ORDER:
- Higher `helpful` score survives
- If tied → keep the lower numeric key (older entry), delete the higher numeric key

## Step 3 — Flag contradictions (do NOT silently resolve them)
When two entries contradict each other, issue exactly 3 tool calls:
1. REMOVE tool call → delete entry [id_A]
2. REMOVE tool call → delete entry [id_B]
3. ADD tool call → create a new entry in section "Contradictions" with content:
   "CONFLICT — [id_A] stated: '[content_A]' | [id_B] stated: '[content_B]'
   — Resolution required."

Do not decide which entry is correct. Do not keep either original entry.

## Step 4 — Improve clarity (optional, conservative)
When a single entry is genuinely vague or unclear, issue exactly 1 tool call:
1. UPDATE tool call → rewrite the entry to be specific and actionable

Only rewrite if truly necessary. Preserve specific wording and numbers exactly.
Do not merge entries that are independent just to reduce count.

# Critical Constraints
- Work ONLY with existing content — do not invent new facts or guidance
- Duplicate detection is SEMANTIC, not textual — rephrasings and cross-section
  duplicates must be caught
- For contradictions: flag via tool calls, never silently resolve
- You MUST issue ALL required tool calls in a single response turn
- Do not wait for tool results before issuing the next tool call

# Output Format
- Use ONLY tool calls to update the playbook
- No explanatory text outside tool calls
- Each tool call must be valid JSON
- Complete all updates in a single response
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