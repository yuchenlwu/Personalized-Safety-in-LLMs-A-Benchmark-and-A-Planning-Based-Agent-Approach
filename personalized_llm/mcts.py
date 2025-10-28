
"""
A simple UCT-based MCTS for attribute selection.
Each action is asking one remaining attribute; a rollout samples its value,
updates the state, and obtains a reward from the judge. Depth limited.
"""
from __future__ import annotations
import math
import random
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Tuple
from .env import QueryState, AttributeSpace
from .user_simulator import simulate_attribute_value
from .judge import evaluate_reward
from . import config
from .llm_client import LLMClient

random.seed(config.SEED)

@dataclass
class Node:
    state: QueryState
    parent: Optional["Node"] = None
    parent_action: Optional[str] = None
    children: Dict[str, "Node"] = field(default_factory=dict)
    N: int = 0  # visit count
    Q: float = 0.0  # total value

    def is_leaf(self) -> bool:
        return len(self.children) == 0

class MCTS:
    def __init__(self, space: AttributeSpace, user_query: str, llm: LLMClient):
        self.space = space
        self.user_query = user_query
        self.llm = llm

    def uct(self, node: Node, action: str, c: float) -> float:
        child = node.children.get(action)
        if child is None or child.N == 0:
            return float("inf")
        return (child.Q / child.N) + c * math.sqrt(math.log(node.N + 1) / child.N)

    def select(self, node: Node, c: float):
        """Traverse the tree until a leaf or depth limit; expand if needed."""
        path = []
        depth = 0
        while depth < config.MCTS_MAX_DEPTH:
            rem = node.state.remaining(self.space)
            if not rem:
                break
            # If any unexpanded action exists, stop and let expand() handle it.
            unexpanded = [a for a in rem if a not in node.children]
            if unexpanded:
                break
            # otherwise pick best by UCT among expanded actions
            best_a = max(rem, key=lambda a: self.uct(node, a, config.MCTS_UCT_C))
            node = node.children[best_a]
            path.append(best_a)
            depth += 1
        return node, path

    def expand(self, node: Node) -> Optional[str]:
        rem = node.state.remaining(self.space)
        unexpanded = [a for a in rem if a not in node.children]
        if not unexpanded:
            return None
        action = random.choice(unexpanded)
        # simulate attribute value to create child state
        val = simulate_attribute_value(action, node.state, self.llm)
        child_state = node.state.apply(action, val)
        child = Node(state=child_state, parent=node, parent_action=action)
        node.children[action] = child
        return action

    def rollout(self, node: Node) -> float:
        """Perform a simulated trajectory from this node to budget, then judge."""
        s = node.state
        steps = 0
        while steps < config.INTERACTION_BUDGET and len(s.known) < len(self.space.attributes):
            rem = [a for a in self.space.attributes if a not in s.known]
            if not rem:
                break
            a = random.choice(rem)
            v = simulate_attribute_value(a, s, self.llm)
            s = s.apply(a, v)
            steps += 1
        return evaluate_reward(self.user_query, s, self.llm)

    def backprop(self, node: Node, value: float):
        while node is not None:
            node.N += 1
            node.Q += value
            node = node.parent

    def best_action(self, root: Node) -> str:
        rem = root.state.remaining(self.space)
        if not rem:
            return ""
        best, best_val = None, -1.0
        for a in rem:
            child = root.children.get(a)
            avg = (child.Q / child.N) if child and child.N > 0 else -1.0
            if avg > best_val:
                best, best_val = a, avg
        return best or rem[0]

    def search(self, init_state: QueryState):
        root = Node(state=init_state)
        for _ in range(config.MCTS_NUM_ROLLOUTS):
            leaf, _ = self.select(root, config.MCTS_UCT_C)
            a = self.expand(leaf)
            node_for_rollout = leaf.children.get(a) if a else leaf
            value = self.rollout(node_for_rollout)
            self.backprop(node_for_rollout, value)
        # Extract greedy path up to budget
        path = []
        cur = root
        depth = 0
        while depth < config.INTERACTION_BUDGET:
            a = self.best_action(cur)
            if not a or a not in cur.children:
                break
            cur = cur.children[a]
            path.append(a)
            depth += 1
        # Final state reward
        final_reward = evaluate_reward(self.user_query, cur.state if cur else root.state, self.llm)
        return path, final_reward
