
"""
Environment and state tracking for attribute acquisition.
A 'state' is the set of known (attribute -> value) pairs.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List

@dataclass
class AttributeSpace:
    """Holds the candidate attributes to ask about."""
    attributes: List[str]

@dataclass
class QueryState:
    """Current known info collected from the user (or simulator)."""
    known: Dict[str, str] = field(default_factory=dict)

    def is_known(self, attr: str) -> bool:
        return attr in self.known

    def remaining(self, space: AttributeSpace) -> List[str]:
        return [a for a in space.attributes if a not in self.known]

    def apply(self, attr: str, value: str) -> "QueryState":
        new_known = dict(self.known)
        new_known[attr] = value
        return QueryState(known=new_known)

    def as_prompt_fragment(self) -> str:
        if not self.known:
            return "No user background is known yet."
        parts = [f"{k}: {v}" for k, v in self.known.items()]
        return "Known user background:\n" + "\n".join(parts)
