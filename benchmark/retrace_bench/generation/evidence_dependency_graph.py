from dataclasses import dataclass, field
from typing import Dict, List, Set


@dataclass
class DependencyNode:
    event_id: str
    requires_event_ids: Set[str] = field(default_factory=set)
    blocks_event_ids: Set[str] = field(default_factory=set)
    releases_event_ids: Set[str] = field(default_factory=set)


class EvidenceDependencyGraph:
    """Manages explicit event dependencies to solve for the minimal gold evidence path."""

    def __init__(self):
        self.nodes: Dict[str, DependencyNode] = {}

    def add_event(self, event_id: str):
        if event_id not in self.nodes:
            self.nodes[event_id] = DependencyNode(event_id)

    def add_requires(self, from_event: str, to_event: str):
        self.add_event(from_event)
        self.add_event(to_event)
        self.nodes[from_event].requires_event_ids.add(to_event)

    def add_blocks(self, from_event: str, to_event: str):
        self.add_event(from_event)
        self.add_event(to_event)
        self.nodes[from_event].blocks_event_ids.add(to_event)

    def add_releases(self, from_event: str, to_event: str):
        self.add_event(from_event)
        self.add_event(to_event)
        self.nodes[from_event].releases_event_ids.add(to_event)

    def compute_minimal_evidence(self, target_event_id: str) -> Set[str]:
        """BFS/DFS to collect all ancestor required events."""
        visited: Set[str] = set()
        queue: List[str] = [target_event_id]
        
        while queue:
            curr = queue.pop(0)
            if curr in visited:
                continue
            visited.add(curr)
            if curr in self.nodes:
                for req in self.nodes[curr].requires_event_ids:
                    queue.append(req)
                for blk in self.nodes[curr].blocks_event_ids:
                    queue.append(blk)
                for rel in self.nodes[curr].releases_event_ids:
                    queue.append(rel)
        return visited
