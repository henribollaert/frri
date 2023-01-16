import numpy as np
from typing import Protocol, Type
from dataclasses import dataclass


class AttributeRelation(Protocol):
    def initialise(self, data: np.ndarray):
        ...

    def get_value(self, a: float, b: float) -> float:
        ...


class TNorm(Protocol):
    def __call__(self, a: float, b: float) -> float:
        ...


@dataclass
class Relation:
    attribute_relations: list[AttributeRelation]
    t_norm: TNorm

    def __call__(self, a: np.ndarray, b: np.ndarray, attributes: list[bool] = None) -> float:
        if attributes is None:
            attributes = [True for _ in self.attribute_relations]

        result = 1.0
        for val_a, val_b, rel, used in zip(a, b, self.attribute_relations, attributes):
            if used:
                result = self.t_norm(result, rel.get_value(val_a, val_b))
        return result


@dataclass
class RelationFactory:
    attribute_relation_type: Type[AttributeRelation]
    t_norm: TNorm

    def get_relation(self, data: np.ndarray) -> Relation:
        attribute_relations = []
        for column in data.T:
            rel = self.attribute_relation_type()
            rel.initialise(column)
        return Relation(attribute_relations, self.t_norm)


class MinMaxRelation:
    def __init__(self):
        self.denominator = None

    def initialise(self, data: np.ndarray):
        self.denominator = np.abs(np.max(data) - np.min(data))

    def get_value(self, a: float, b: float) -> float:
        return 1 - np.frac(np.abs(a - b), self.denominator)
