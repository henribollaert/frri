from dataclasses import dataclass
from quickrules.fuzzy_set import FuzzySet
import numpy as np
from typing import Protocol, Optional, TypeVar
import math

LabelType = TypeVar('LabelType', int, str)


class TNorm(Protocol):
    def __call__(self, a: float, b: float) -> float:
        ...


class Implicator(Protocol):
    def __call__(self, a: float, b: float) -> float:
        ...


class Relation(Protocol):

    def __call__(self, a: np.ndarray, b: np.ndarray, attributes: list[bool]) -> float:
        ...


class RelationFactory(Protocol):

    def get_relation(self, data: np.ndarray) -> Relation:
        ...


@dataclass
class Rule:
    """
    This class represents a single rule. It can be defined by the attributes that are part of the rule and the
    generating element, from which we can determine the defining attribute-value pairs of the rule.
    Finally, we also need the decision value of the rule.

    This class is a pure data class without any function. Evaluating a rule and calculating
    its coverage is left to the QuickRules class.
    """
    attributes: list[bool]  # set of conditional attributes in the rule's antecedent
    generating_element: np.ndarray  # fuzzy tolerance class of the object that generated the rule
    decision: LabelType  # the decision class/consequent of the rule. type depends on the data set
    coverage: FuzzySet

    def __str__(self):
        s = "("
        for nr, (used, value) in enumerate(zip(self.attributes, self.generating_element)):
            if used:
                s += f"a_{nr+1}={value},"
        s += f"decision={self.decision})"
        return s

    def __eq__(self, other):
        if not isinstance(other, Rule):
            return False
        if other is self:
            return True
        if self.attributes != other.attributes:
            return False
        if not np.array_equal(self.generating_element, other.generating_element):
            return False
        if self.decision != other.decision:
            return False
        return True


class QuickRules:
    """

    This class needs an attribute based relation, since it is needed in the calculation of the POS region.
    """

    def __init__(
            self,
            t_norm: TNorm,
            implicator: Implicator,
            relation_factory: RelationFactory
    ):
        self.t_norm: TNorm = t_norm
        self.implicator: Implicator = implicator
        self.relation_factory: RelationFactory = relation_factory
        self.relation: Optional[Relation] = None

        self.rules: Optional[list[Rule]] = None
        self.covered: Optional[FuzzySet] = None

        self.nr_of_attributes: Optional[int] = None
        self.X: Optional[np.ndarray] = None
        self.y: Optional[np.ndarray] = None

    def get_info(self):
        return f"@t-norm: {self.t_norm}\n@implicator: {self.implicator}\n@rel:{self.relation_factory}\n"

    def get_rules_as_string(self) -> list[str]:
        return [str(rule) for rule in self.rules]

    def fit(self, x: np.ndarray, y: np.ndarray):  # should just use the quick rules algorithm
        # save input
        self.X = x
        self.nr_of_attributes = x.shape[1]
        self.y = y

        self.relation = self.relation_factory.get_relation(x)

        # initialisation of quick rules
        self.rules = []
        used_attributes = [False for _ in range(self.nr_of_attributes)]
        self.covered = FuzzySet()

        pos_a = self._get_positive_region()  # calculate the positive region for all attributes
        gamma_a = pos_a.get_size()  # we use the numerator, since denominator is the same everywhere
        gamma_b = 0

        # main loop
        while not math.isclose(gamma_b, gamma_a):
            # temporary additions to B
            temp_gamma = gamma_b
            best_attribute = None
            for attribute in range(self.nr_of_attributes):

                # we skip the already used attributes
                if used_attributes[attribute]:
                    continue

                b_with_a = [_ for _ in used_attributes]
                b_with_a[attribute] = True

                gamma_b_with_a = 0.0  # we will need to compare this later, and don't need a fuzzy set for this
                for sample_index, (sample, label) in enumerate(zip(self.X, self.y)):
                    membership = self._calculate_single_pos_membership(sample, label, b_with_a)
                    gamma_b_with_a += membership

                    # we skip fully covered samples
                    if math.isclose(self.covered.get_membership(sample_index),
                                        pos_a.get_membership(sample_index)):
                        continue

                    if math.isclose(membership, pos_a.get_membership(sample_index)):
                        new_rule = self._create_rule(b_with_a, sample, label)
                        self.check(new_rule)
                # print(gamma_b_with_a, temp_gamma)
                if gamma_b_with_a > temp_gamma:
                    # print('larger')
                    best_attribute = attribute
                    temp_gamma = gamma_b_with_a
            # update b with the best one we found
            used_attributes[best_attribute] = True
            gamma_b = temp_gamma

    def check(self, rule_to_check: Rule):
        # first we check if the new rule will increase the coverage of at least one element
        add = True
        to_remove = []
        for rule in self.rules:
            if rule_to_check.coverage.is_subset_of(rule.coverage):
                add = False
                break
            elif rule.coverage.is_subset_of(rule_to_check.coverage):
                to_remove.append(rule)
        if add:
            # if we will add, we remove any rules of which the coverage is a subset of the new rule
            for rule in to_remove:
                self.rules.remove(rule)
            # then we add the rule and update the coverage
            self.rules.append(rule_to_check)
            self.covered = self.covered.union(rule_to_check.coverage)

    def _create_rule(self, attributes, generating_element, decision) -> Rule:
        """
        Creates a rule defined by the given attributes, generating element and decision class.
        We calculate the coverage of the rule in the training set during creation
        :param attributes:  attributes used in the antecedent of the rule
        :param generating_element:  element which holds the values for those attributes
        :param decision:  decision class of the consequent
        :return:  a new rule
        """
        # first we calculate the coverage
        coverage = FuzzySet()
        for index, sample in enumerate(self.X):
            membership = self.relation(generating_element, sample, attributes)
            if membership > 0.0:
                coverage.add(index, membership)
        # now we have all required parameters
        return Rule(attributes, generating_element, decision, coverage)

    def _evaluate_rule(self, sample, rule):
        return self.relation(sample, rule.generating_element, rule.attributes)

    def _predict_single(self, sample: np.ndarray) -> LabelType:  # evaluate rules
        max_value = 0.0
        pred = None
        for rule in self.rules:
            if temp := self._evaluate_rule(sample, rule) > max_value:
                max_value = temp
                pred = rule.decision
        return pred

    def _predict_single_combo(self, sample: np.ndarray) -> LabelType:
        preds = {}
        for rule in self.rules:
            preds[rule.decision] = preds.get(rule.decision, 0.0) + self._evaluate_rule(sample, rule)
        return max(preds, key=preds.get, default=None)

    # todo we might want the option to get the firing values for each rule
    def predict(self, x: np.ndarray) -> np.ndarray:
        # result = np.array([self._predict_single(sample) for sample in x])
        # return result
        return np.array([self._predict_single_combo(sample) for sample in x])

    def _get_positive_region(self, attributes=None) -> FuzzySet:
        """
        Calculates the positive region for the given set of attributes. If attributes is none,
        all attributes are used.
        :param attributes:  boolean array corresponding to the selected attributes
        """
        if attributes is None:
            attributes = [True] * self.nr_of_attributes

        pos = FuzzySet()
        for index, (sample, label) in enumerate(zip(self.X, self.y)):
            pos.add(index, self._calculate_single_pos_membership(sample, label, attributes))
        return pos

    def _calculate_single_pos_membership(self,
                                         sample: np.ndarray,
                                         label: LabelType,
                                         attributes: list[bool] = None) -> float:
        """
        Calculates POS_attributes(sample)
        :param sample: ndarray containing the values of the conditional attributes of the sample
        :param label: decision attribute of the sample
        :param attributes: a list of booleans containing True if we use the attribute. If None, we use all attributes
        :return: the membership degree to the positive region given by the attributes for the sample.
        """
        if attributes is None:
            attributes = [True] * self.nr_of_attributes

        membership = 1
        for other, other_label in zip(self.X, self.y):
            membership = min(membership,
                             self.implicator(self.relation(sample,
                                                           other,
                                                           attributes),
                                             int(label == other_label)))
        return membership
