from abc import ABC, abstractmethod

class Prompts(ABC):
    
    def __init__(self, policyDb, hypothesesDb):
        self.policyDb = policyDb
        self.hypothesesDb = hypothesesDb
    
    @abstractmethod
    def getGeneratorPrompt(self, initialState) -> str:
        pass
    
    @abstractmethod
    def getGeneratorTools(self) -> str:
        pass
    
    @abstractmethod
    def getReflectorPrompt(self, trajectory) -> str:
        pass
    
    @abstractmethod
    def getCuratorPrompt(self, trajectory, reflection) -> str:
        pass
    
    @abstractmethod
    def getCuratorTools(self) -> str:
        pass