from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Iterable, Optional, Dict, Any

from .agents import Agent, Env
from .reasoning import ReasoningBase
from .memory import CoMem, MASMemoryBase

@dataclass
class MetaMAS:
    """Unified management of members in the MAS system to achieve overall scheduling of the MAS.
    """
    agents_team: Dict[str, Agent] = field(default_factory=dict)  
    env: Optional[Env] = None  
    meta_memory: Optional[MASMemoryBase] = None  
    
    def hire(self, agents: Iterable[Agent]) -> None:
        for agent in agents:
            if agent.name not in self.agents_team:
                self.agents_team[agent.name] = agent  
            else:
                print(f"Agent {agent.name} is already in the team.")  

    def set_env(self, env: Env) -> None:
        self.env = env

    def set_memory(self, memory, *auxiliary_agents):
        self.meta_memory = memory
        if isinstance(memory, CoMem):
            memory.bind((*self.agents_team.values(), *auxiliary_agents))

    def save_episode(self, reward, done, feedback):
        kwargs = {'reward': reward} if isinstance(self.meta_memory, CoMem) else {}
        self.meta_memory.save_task_context(label=done, feedback=feedback, **kwargs)
    
    def get_agent(self, agent_name: str) -> Optional[Agent]:
        return self.agents_team.get(agent_name)
    
    @abstractmethod
    def build_system(self, reasoning: ReasoningBase, mas_memory: MASMemoryBase) -> Any:
        pass
    
    @abstractmethod
    def schedule(self, **args) -> Any:
        pass
