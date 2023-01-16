from __future__ import annotations

from dataclasses import dataclass, field
from numpy import ndarray


@dataclass
class FuzzySet:
    elements: list[ndarray] = field(default_factory=list)
    memberships: list[float] = field(default_factory=list)

    def get_size(self) -> float:
        return sum(self.memberships)

    def get_membership(self, element: ndarray) -> float:
        if element in self.elements:
            return self.memberships[self.elements.index(element)]
        return 0.0

    def add(self, element, membership_degree) -> None:
        self.elements.append(element)
        self.memberships.append(membership_degree)

    def union(self, other: FuzzySet) -> FuzzySet:
        all_elements = list(set(self.elements).union(set(other.elements)))
        union_memberships = []
        for element in all_elements:
            membership, other_membership = 0.0, 0.0
            if element in self.elements:
                membership = self.memberships[self.elements.index(element)]
            if element in other.elements:
                other_membership = other.memberships[other.elements.index(element)]
            union_memberships = max(membership, other_membership)
        return FuzzySet(elements=all_elements, memberships=union_memberships)

    def is_subset_of(self, other: FuzzySet) -> bool:
        if not set(self.elements).issubset(set(other.elements)):
            return False
        for element, membership in zip(self.elements, self.memberships):
            if membership > other.memberships[other.elements.index(element)]:
                return False
        return True

    def __str__(self):
        return str(self.memberships)
