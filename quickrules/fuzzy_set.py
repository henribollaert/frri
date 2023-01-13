from __future__ import annotations

from dataclasses import dataclass, field
from numpy import ndarray


@dataclass
class FuzzySet:
    memberships: dict[ndarray, float] = field(default_factory=dict)

    def get_size(self) -> float:
        return sum(self.memberships.values())

    def get_membership(self, element: ndarray) -> float:
        return self.memberships.get(element, 0.0)

    def add(self, element, membership_degree) -> None:
        self.memberships[element] = membership_degree

    def union(self, other: FuzzySet) -> FuzzySet:
        elements = set(self.memberships.keys()).union(set(other.memberships.keys()))
        union_memberships = {}
        for element in elements:
            union_memberships = max(self.memberships.get(element, 0.0), other.memberships.get(element, 0.0))
        return FuzzySet(memberships=union_memberships)

    def is_subset_of(self, other: FuzzySet) -> bool:
        if not set(self.memberships.keys()).issubset(other.memberships.keys()):
            return False
        for element, membership in self.memberships.items():
            if membership > other.memberships[element]:
                return False
        return True
