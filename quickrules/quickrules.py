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
    """
    A relation takes two samples and a list of booleans, where True indicates that the corresponding
    attribute should be used.
    It then returns the degree of relation of those two samples w.r.t. those attributes.
    """
    def __call__(self, a: np.ndarray, b: np.ndarray, attributes: list[bool]) -> float:
        ...


class RelationFactory(Protocol):
    """
    A RelationFactory returns a Relation that is normalized for the input data.
    """
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
    generating_element: np.ndarray  # the element that generated this rule
    decision: LabelType  # the decision class/consequent of the rule. type depends on the data set
    coverage: FuzzySet  # fuzzy tolerance class of the object that generated the rule

    def __str__(self):
        s = "("
        for nr, (used, value) in enumerate(zip(self.attributes, self.generating_element)):
            if used:
                s += f"a_{nr + 1}={value},"
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
    Implementation of the QuickRules algorithm from [1], with an added pruning step.

    [1] Jensen, Richard & Cornelis, Chris & Shen, Qiang. (2009). Hybrid Fuzzy-Rough Rule Induction and
        Feature Selection. IEEE International Conference on Fuzzy Systems. 1151 - 1156. 10.1109/FUZZY.2009.5277058.
    """

    def __init__(
            self,
            t_norm: TNorm,
            implicator: Implicator,
            relation_factory: RelationFactory,
            prune: bool = False,
            combo: bool = True
    ):
        """
        Initialisation of a QuickRules object. Sets the hyperparameters of the algorithm
        :param t_norm: t-norm used in aggregation of the attribute-based relation and the # todo wordt eigenlijk niet gebruikt
        :param implicator: implicator used for calculating the lower approximation
        :param relation_factory: factory that can return relation objects that take into account the training data
        :param prune: toggles pruning on or off
        :param combo: selects the prediction method
        """
        self.t_norm: TNorm = t_norm
        self.implicator: Implicator = implicator
        self.relation_factory: RelationFactory = relation_factory
        self.relation: Optional[Relation] = None
        self.prune: bool = prune
        self.combo: bool = combo

        self.rules: Optional[list[Rule]] = None
        self.covered: Optional[FuzzySet] = None

        self.nr_of_attributes: Optional[int] = None
        self.X: Optional[np.ndarray] = None
        self.y: Optional[np.ndarray] = None

    def get_info(self) -> str:
        return f"@t-norm: {self.t_norm}\n@implicator: {self.implicator}\n@rel:{self.relation_factory}\n"

    def get_rules_as_string(self) -> list[str]:
        return [str(rule) for rule in self.rules]

    def _init_fit(self, x: np.ndarray, y: np.ndarray, t: np.ndarray) -> None:
        """
        Initialises the data containing members and relation of this object before starting the rule induction.
        :param x: numerical data
        :param y: labels
        :param t: datatypes of each feature
        :return: nothing
        """
        self.X = x
        self.nr_of_attributes = x.shape[1]
        self.y = y

        self.relation = self.relation_factory.get_relation(x, t)

    def fit(self, x: np.ndarray, y: np.ndarray, t: np.ndarray) -> None:
        """
        Creates a set of rules using the QuickRules algorithm on the data set (x,y), where x contains the
        conditional attributes of the samples and y contains their labels.
        :param x: numerical data
        :param y: labels
        :param t: datatypes of each feature
        :return: nothing
        """
        self._init_fit(x, y, t)

        # initialisation of quick rules
        self.rules = []
        used_attributes = [False for _ in range(self.nr_of_attributes)]
        self.covered = FuzzySet()

        pos_a = self._get_positive_region()  # calculate the positive region for all attributes
        gamma_a = pos_a.get_size()  # we use the numerator, since denominator is the same everywhere
        gamma_b = 0

        # main loop
        while not math.isclose(gamma_b, gamma_a):
            # temporary additions to B: we will greedily look for the best attribute to add
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
                if gamma_b_with_a > temp_gamma:
                    best_attribute = attribute
                    temp_gamma = gamma_b_with_a

            # update b with the best attribute, if possible
            if best_attribute is None:
                print(f"No best attribute found! Gamma_b is {gamma_b}, while Gamma_a is {gamma_a}. Used attributes are:")
                print(used_attributes)
                break

            used_attributes[best_attribute] = True
            gamma_b = temp_gamma

        # after the rule induction with QuickRules, we can end with an optional pruning step.
        if self.prune:
            self._prune()

    def _prune(self) -> None:
        """
        This method prunes the rules after fitting by trying to remove each rule from the
        ruleset and doing so when all elements are still covered.
        """
        to_remove = []
        for rule_to_check in self.rules:
            for i, x in enumerate(self.X):
                for other_rule in self.rules:
                    if rule_to_check == other_rule:
                        continue
                    if math.isclose(other_rule.coverage.get_membership(i), 1.0):  # we use the index!
                        break
                else:
                    # this sample was not covered by any other rule
                    break
            else:
                # the above loop never encountered a break: all samples were fully covered
                # by another rule -> we can remove this rule
                to_remove.append(rule_to_check)

        self.rules = [rule for rule in self.rules if rule not in to_remove]

    def check(self, rule_to_check: Rule) -> None:
        """
        Checks whether we should add the given rule to the set of rules.
        Can also result in the removal of rules if their coverage is a subset of the coverage
        of this rule.
        :param rule_to_check: new candidate rule
        :return: nothing
        """
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
        :param attributes: attributes used in the antecedent of the rule
        :param generating_element: element which holds the values for those attributes
        :param decision: decision class of the consequent
        :return: a new rule
        """
        # first we calculate the coverage
        coverage = FuzzySet()
        for index, sample in enumerate(self.X):
            membership = self.relation(generating_element, sample, attributes)
            if membership > 0.0:
                coverage.add(index, membership)
        # now we have all required parameters
        return Rule(attributes, generating_element, decision, coverage)

    def _evaluate_rule(self, sample, rule) -> float:
        """
        Evaluates a given rule on a given sample by calculating the firing value of the rule, which is
        given by the relationship degree between the generating element of the rule and the new sample w.r.t.
        the attributes of the rule.
        :param sample: new sample to test
        :param rule: rule to evaluate
        :return: firing value of the rule
        """
        return self.relation(sample, rule.generating_element, rule.attributes)

    def _predict_single(self, sample: np.ndarray) -> LabelType:
        """
        Returns the class with the highest firing rule for a given sample.
        :param sample: np.array containing the conditional attributes of the sample
        :return: prediction
        """
        predictions = self._predict_proba_single(sample)
        return max(predictions, key=predictions.get, default=None)

    def _predict_single_combo(self, sample: np.ndarray) -> LabelType:
        """
        Returns the class for which the total sum of all the firing rates of its rules is the highest.
        :param sample: np.array containing the conditional attributes of the sample
        :return: prediction
        """
        predictions = self._predict_proba_single_combo(sample)
        return max(predictions, key=predictions.get, default=None)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Returns the most likely class for each sample in the two-dimensional array x containing the
        conditional attributes of each sample.
        :param x: numpy array of samples
        :return: numpy array of predictions
        """
        return np.array([max(predictions, key=predictions.get, default=None)
                         for predictions in self.predict_proba(x)])

    @staticmethod
    def _normalise_dict(d: dict) -> dict:
        """
        Helper function that normalises the values of a dict such that their sum will be 1.
        :param d: dict to normalise
        :return: normalised dict
        """
        div = sum(d.values())
        if div == 0:  # to avoid diving by 0
            return d
        return {k: v / div for k, v in d.items()}

    def _predict_proba_single(self, sample: np.ndarray) -> dict[(LabelType, float)]:
        """
        Returns probabilities of this sample belonging to each class based on the highest firing rule for that sample.
        :param sample: np.array containing the conditional attributes of the sample
        :return: prediction
        """
        max_values = {}
        for rule in self.rules:
            max_values[rule.decision] = max(max_values.get(rule.decision, 0.0), self._evaluate_rule(sample, rule))
        return self._normalise_dict(max_values)

    def _predict_proba_single_combo(self, sample: np.ndarray) -> dict[(LabelType, float)]:
        """
        Returns the probability of the sample belonging to each class based on the total sum of
        all the firing rates of each rule.
        :param sample: np.array containing the conditional attributes of the sample
        :return: prediction
        """
        total_values = {}
        for rule in self.rules:
            total_values[rule.decision] = total_values.get(rule.decision, 0.0) + self._evaluate_rule(sample, rule)
        return self._normalise_dict(total_values)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """
        Returns a dictionary of classes and probabilities for each sample in the two-dimensional array x of samples.
        :param x: two-dimensional of samples whose classes we want to predict
        :return: np array of dicts, one for each element in x, containing the
                 probability of that element belonging to each class
        """
        return np.array([self._predict_proba_single_combo(sample) if self.combo else self._predict_proba_single(sample)
                         for sample in x])

    def _get_positive_region(self, attributes=None) -> FuzzySet:
        """
        Calculates the positive region for the given set of attributes. If attributes is none,
        all attributes are used.
        :param attributes:  boolean array corresponding to the selected attributes
        :return: the positive region as a fuzzy set, where the samples of x are represented by their index
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
