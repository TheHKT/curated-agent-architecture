from optimization.prompts.Prompts import Prompts
from optimization.policy.Curator import Curator
from optimization.policy.Reflector import Reflector
from optimization.PlaybookRefactorer import PlaybookRefactorer

class PolicyRefiner:
    def __init__(self, client, model, policyDb, prompts: Prompts):
        self.client = client
        self.model = model
        self.policyDb = policyDb
        self.prompts = prompts

    def run(self, trajectory, debug=False): 
        reflector = Reflector(self.client, self.model, self.policyDb, self.prompts)
        reflexion = reflector.run(trajectory, debug=debug)
        
        curator = Curator(self.client, self.model, self.policyDb, self.prompts)
        curation_str = curator.run(trajectory, reflexion, debug=debug)

        playbook_refactorer = PlaybookRefactorer(self.client, self.model, self.policyDb)
        playbook_refactorer.run(debug=debug)

        return curation_str