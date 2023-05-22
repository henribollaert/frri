import numpy as np
from typing import Protocol, Type, TypeVar
from dataclasses import dataclass

AttributeType = TypeVar('AttributeType', float, str)


class AttributeRelation(Protocol[AttributeType]):
    def initialise(self, data: np.ndarray):
        ...

    def get_value(self, a: AttributeType, b: AttributeType) -> float:
        ...


class Aggregator(Protocol):
    def __call__(self, a: float, b: float) -> float:
        ...


@dataclass
class Relation:
    attribute_relations: list[AttributeRelation]
    aggregator: Aggregator

    def __call__(self, a: np.ndarray, b: np.ndarray, attributes: list[bool] = None) -> float:
        if attributes is None:
            attributes = [True for _ in self.attribute_relations]

        to_aggregate = []
        for val_a, val_b, rel, used in zip(a, b, self.attribute_relations, attributes):
            if used:
                to_aggregate.append(rel.get_value(val_a, val_b))
        result = to_aggregate.pop()
        while len(to_aggregate) > 0:
            result = self.aggregator(result, to_aggregate.pop())
        return result


@dataclass
class RelationFactory:
    numerical_relation_type: Type[AttributeRelation[float]]
    categorical_relation_type: Type[AttributeRelation[str]]
    aggregator: Aggregator

    def get_relation(self, data: np.ndarray, types: np.ndarray) -> Relation:
        attribute_relations = []
        for column, t in zip(data.T, types):
            rel = self.numerical_relation_type() \
                if t == np.dtype('float64') else self.categorical_relation_type()
            rel.initialise(column)
            attribute_relations.append(rel)
        return Relation(attribute_relations, self.aggregator)


@dataclass
class AverageRelation:
    """
    This relation uses the monotonic average trick. The normal average is not monotonic and thus
    can decrease when considering more attributes.
    """
    attribute_relations: list[AttributeRelation]

    def __call__(self, a: np.ndarray, b: np.ndarray, attributes: list[bool] = None) -> float:
        if attributes is None:
            attributes = [True for _ in self.attribute_relations]

        to_aggregate = []
        for val_a, val_b, rel, used in zip(a, b, self.attribute_relations, attributes):
            if used:
                to_aggregate.append(rel.get_value(val_a, val_b))
            else:
                to_aggregate.append(1)
        return np.average(to_aggregate)


@dataclass
class AverageRelationFactory:
    numerical_relation_type: Type[AttributeRelation[float]]
    categorical_relation_type: Type[AttributeRelation[str]]

    def get_relation(self, data: np.ndarray, types: np.ndarray) -> AverageRelation:
        attribute_relations = []
        for column, t in zip(data.T, types):
            rel = self.numerical_relation_type() \
                if t == np.dtype('float64') else self.categorical_relation_type()
            rel.initialise(column)
            attribute_relations.append(rel)
        return AverageRelation(attribute_relations)


class MinMaxRelation:
    def __init__(self):
        self.denominator = None

    def initialise(self, data: np.ndarray):
        self.denominator = np.abs(np.max(data) - np.min(data))

    def get_value(self, a: float, b: float) -> float:
        return 1 - np.divide(np.abs(a - b), self.denominator)


class StatisticalRelation:
    def __init__(self):
        self.std_dev = None

    def initialise(self, data: np.ndarray):
        self.std_dev = np.std(data)

    def get_value(self, a: float, b: float) -> float:
        return max(min((a - b)/self.std_dev, (b - a)/self.std_dev) + 1, 0)

    def __repr__(self):
        return "Statistical relation"


class CategoricalRelation:
    def initialise(self, data: np.ndarray):
        return

    def get_value(self, a: str, b: str) -> float:
        return int(a == b)
