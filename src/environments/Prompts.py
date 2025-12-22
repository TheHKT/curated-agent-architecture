from abc import ABC, abstractmethod

class Prompts(ABC):
    @abstractmethod
    def getGeneratorPrompt(self) -> str:
        pass
    
    @abstractmethod
    def getGeneratorTools(self) -> str:
        pass
    
    @abstractmethod
    def getReflectorPrompt(self) -> str:
        pass
    
    @abstractmethod
    def getCuratorPrompt(self) -> str:
        pass
    
    @abstractmethod
    def getCuratorTools(self) -> str:
        pass