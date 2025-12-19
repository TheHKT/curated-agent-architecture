from abc import ABC, abstractmethod
from typing import Dict, Callable, Any

class Environment(ABC):
    """
    Abstract base class for gymnasium environment wrappers.
    Each subclass must define its own Action enum.
    """
    
    def __init__(self, env):
        self.env = env
    
    @property
    @abstractmethod
    def ACTION_MAP(self) -> Dict[str, Callable]:
        """
        Map tool names to their corresponding functions.
        Must be implemented by subclasses.
        """
        pass
    
    @abstractmethod
    def getState(self) -> str:
        """
        Get a human-readable description of the current game state.
        This is used to represent the state for an LLM or human observer.
        """
        pass
    
    @abstractmethod
    def executeAction(self, action: Any) -> Dict[str, Any]:
        """
        Execute the given action in the environment.

        Args:
            action: The action to execute, typically an enum member.

        Returns:
            Dict[str, Any]: A dictionary containing the new state, reward, isTerminated flag
        """
        pass
    
    @abstractmethod
    def getInitialGeneratorPrompt(self, policy, hyptothesisDb) -> str:
        pass
    
    @abstractmethod
    def getGeneratorTools(self) -> str:
        pass
    
    def reset(self, seed=None):
        """Reset the environment."""
        return self.env.reset(seed=seed)
    
    def close(self):
        """Close the environment."""
        self.env.close()