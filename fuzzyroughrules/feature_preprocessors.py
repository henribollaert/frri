"""
This file contains functions that preprocess
the order of a set of features for a given dataset.
"""

from typing import Optional
import numpy as np
from math import isclose

from fuzzyroughrules.operators import lukasiewicz_implicator
from quickrules.quickrules import Relation, LabelType
from quickrules.relations import MinMaxRelation, CategoricalRelation, RelationFactory
from quickrules.fuzzy_operators import MinTNorm


class QuickReduct:

    def __init__(
            self,
            implicator=lukasiewicz_implicator,
            relation_factory: RelationFactory = RelationFactory(MinMaxRelation, CategoricalRelation, MinTNorm()),
            verbose: bool = False
    ):
        self.implicator = implicator
        self.relation_factory: RelationFactory = relation_factory
        self.verbose: bool = verbose

        self.relation: Optional[Relation] = None
        self.X = None
        self.nr_of_attributes = None
        self.y = None

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

    def order_features(self, x: np.ndarray, y: np.ndarray, t: np.ndarray) -> np.ndarray[int]:
        self._init_fit(x, y, t)

        new_order = []
        used_attributes = [False for _ in range(self.nr_of_attributes)]

        # calculate the positive region for all attributes
        gamma_a = self._get_positive_region_size()  # we use the numerator, since denominator is the same everywhere
        gamma_b = 0

        # main loop
        while not isclose(gamma_b, gamma_a):
            # temporary additions to B: we will greedily look for the best attribute to add
            temp_gamma = gamma_b
            best_attribute = None
            for attribute in range(self.nr_of_attributes):
                # we skip the already used attributes
                if used_attributes[attribute]:
                    continue

                b_with_a = [_ for _ in used_attributes]
                b_with_a[attribute] = True

                gamma_b_with_a = self._get_positive_region_size(b_with_a)

                if gamma_b_with_a > temp_gamma:
                    best_attribute = attribute
                    temp_gamma = gamma_b_with_a

            # update b with the best attribute, if possible
            if best_attribute is None:
                # todo fix "AttributeError: 'QuickReduct' object has no attribute 'verbose'"
                # if self.verbose:
                print(f"No best attribute found! Gamma_b is {gamma_b},"
                      f" while Gamma_a is {gamma_a}. Used attributes are:")
                print(used_attributes)
                for att, used in enumerate(used_attributes):
                    if not used:
                        new_order.append(att)
                break

            used_attributes[best_attribute] = True
            new_order.append(best_attribute)
            gamma_b = temp_gamma
        return np.array(new_order)

    def _get_positive_region_size(self, attributes: Optional[list[bool]] = None) -> float:
        """
        Calculates the size of the positive region for the given set of attributes.
        If attributes is none, all attributes are used.
        :param attributes:  boolean array corresponding to the selected attributes
        :return: the size of the positive region, where the samples of x are represented by their index
        """
        if attributes is None:
            attributes = [True] * self.nr_of_attributes

        size = 0.0
        for sample, label in zip(self.X, self.y):
            size += self._calculate_single_pos_membership(sample, label, attributes)
        return size

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

        membership = 1  # this is the membership of the sample itself to the lower approx of its class
        for other, other_label in zip(self.X, self.y):
            membership = min(membership,
                             self.implicator(self.relation(sample,
                                                           other,
                                                           attributes),
                                             int(label == other_label)))
        return membership
