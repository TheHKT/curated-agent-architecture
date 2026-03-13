from optimization.prompts.Prompts import Prompts
from tinydb.operations import increment
import json
from tinydb import Query

from utils.util import call_llm

class Reflector:
    def __init__(self, client, model, policyDb, prompts: Prompts):
        self.client = client
        self.model = model
        self.policyDb = policyDb
        self.prompts = prompts
        self.query = Query()
        
    def run(self, trajectory, debug=False): 
        contextMessage = self.prompts.getReflectorPrompt(trajectory)

        msg = call_llm(self.client, self.model, contextMessage, debug=debug)
        if msg and msg.content:
            self.updateTags(msg.content, debug=debug)
        else:
            print("No content returned from LLM in Reflector.run()")

        if debug:
            print(f"== Response: {msg.content}")

        return msg.content
    
    
    
    def updateTags(self, response, debug=False):
        try:
            response_data = json.loads(response)
        except json.JSONDecodeError:
            if debug:
                print("Failed to parse response as JSON")
            return
        
        for entry in response_data.get("bullet_tags", []):
            bullet_id = entry.get("id")
            tag = entry.get("tag")
            try:
                if bullet_id is not None:
                    if tag == "helpful":
                        self.policyDb.update(increment('helpful'), self.query.id == bullet_id)
                    elif tag == "harmful":
                        self.policyDb.update(increment('harmful'), self.query.id == bullet_id)
            except Exception as e:
                if debug:
                    print(f"Error updating tags for bullet_id {bullet_id}: {e}")