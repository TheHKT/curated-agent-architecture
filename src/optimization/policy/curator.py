import json
import uuid 
from optimization.prompts.Prompts import Prompts
from tinydb import Query
from utils.util import execute_tool_call, call_llm

class Curator:
    def __init__(self, client, model, policyDb, prompts: Prompts):
        self.client = client
        self.model = model
        self.policyDb = policyDb
        self.prompts = prompts
        self.query = Query()
        
        self.TOOL_MAPPING = {
          "ADD": self.add,
          "REMOVE": self.remove,
          "MODIFY": self.modify
      }

    def run(self, trajectory, reflexion, debug=False): 
        contextMessage = self.prompts.getCuratorPrompt(trajectory, reflexion)
        
        msg = call_llm(self.client, self.model, contextMessage, self.prompts.getCuratorTools(), debug)
            
        if msg and msg.tool_calls:
            if(debug):
                print(f"== Num_Tools: {len( msg.tool_calls)}")          
            for tool_call in msg.tool_calls:
                tool_response = execute_tool_call(self.TOOL_MAPPING, tool_call, debug)

        return msg.content
    
    
    def add(self, section, content):
        newEntry = {"id": str(uuid.uuid4()), "section": section, "content": content, "helpful": 0, "harmful": 0}
        self.policyDb.insert(newEntry)
    def modify(self, bullet_id, content):
        self.policyDb.update({"content": content}, self.query.id == bullet_id)
    def remove(self, bullet_id):
        self.policyDb.remove(self.query.id == bullet_id)