from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class FuzzySet:
    """
    Class representing a fuzzy set for the quickrules package, where elements are identified by their index in the
    training set.
    """
    memberships: dict[int, float] = field(default_factory=dict)
    t_conorm = max

    def get_size(self) -> float:
        return sum(self.memberships.values())

    def get_membership(self, element: int) -> float:
        return self.memberships.get(element, 0.0)

    def add(self, element: int, membership_degree: float) -> None:
        self.memberships[element] = membership_degree

    def union(self, other: FuzzySet) -> FuzzySet:
        elements = set(self.memberships.keys()).union(set(other.memberships.keys()))
        union_memberships = {}
        for element in elements:
            union_memberships[element] = self.t_conorm(self.memberships.get(element, 0.0),
                                                       other.memberships.get(element, 0.0))
        return FuzzySet(memberships=union_memberships)

    def is_subset_of(self, other: FuzzySet) -> bool:
        if not set(self.memberships.keys()).issubset(other.memberships.keys()):
            return False
        for element, membership in self.memberships.items():
            if membership > other.memberships[element]:
                return False
        return True
