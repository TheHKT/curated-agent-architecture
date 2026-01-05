from prompts.Prompts import Prompts
from tinydb.operations import increment
import json
from tinydb import Query

class Reflector:
    def __init__(self, client, model, policyDb, prompts: Prompts):
        self.client = client
        self.model = model
        self.policyDb = policyDb
        self.prompts = prompts
        self.query = Query()
        
    def run(self, trajectory, debug=False): 
        contextMessage = self.prompts.getReflectorPrompt(trajectory)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=contextMessage
        ).choices[0].message
        
        self.updateTags(response.content, debug=debug)

        if debug:
            print(f"== Response: {response.content}")

        return response.content
    
    
    
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
            
            if bullet_id is not None:
                if tag == "helpful":
                    self.policyDb.update(increment('helpful'), self.query.id == bullet_id)
                elif tag == "harmful":
                    self.policyDb.update(increment('harmful'), self.query.id == bullet_id)