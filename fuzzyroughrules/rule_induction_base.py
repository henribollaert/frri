from typing import Protocol
from dataclasses import dataclass, field

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.preprocessing import normalize, MinMaxScaler
import gurobipy as gb
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from fuzzyroughrules.operators import RelationTypes, triangular_relation, ImplicatorInclusion
from fuzzyroughrules.approximations import LowerApproximation
from fuzzyroughrules.feature_preprocessors import QuickReduct

np.set_printoptions(formatter={'float': lambda x: "{0:0.5f}".format(x)})

inf_val = float('Inf')
import fuzzyroughrules.fuzzy_operators as fo
import importlib

importlib.reload(fo)


class Rule:
    def __init__(self, size):
        self.size = size
        self.left_bounds = inf_val * np.ones(self.size)
        self.right_bounds = inf_val * np.ones(self.size)
        self.condition_json = []
        self.decision = 0

    # todo fix: use RelationTypes here
    def add_condition(self, attribute, type, left_bound, right_bound):
        if type == 'l':
            self.left_bounds[attribute] = left_bound
            pair = {"attribute": attribute, "type": type, "bounds": [left_bound, ]}
            self.condition_json.append(pair)
        if type == 'r':
            self.right_bounds[attribute] = right_bound
            pair = {"attribute": attribute, "type": type, "bounds": [right_bound, ]}
            self.condition_json.append(pair)
        if type == 'i':
            self.left_bounds[attribute] = left_bound
            self.right_bounds[attribute] = right_bound
            pair = {"attribute": attribute, "type": type, "bounds": [left_bound, right_bound]}
            self.condition_json.append(pair)

    def add_decision(self, decision):
        self.decision = decision


class Approximation(Protocol):
    def get_approximation(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        ...


class InclusionMeasure(Protocol):
    def inclusion(self, A: np.ndarray, B: np.ndarray) -> float:
        ...


class FeatureOrdering(Protocol):
    def order_features(self, x: np.ndarray, y: np.ndarray, t: np.ndarray) -> np.ndarray[int]:
        ...


# todo add parameters for t-norm and relation
@dataclass
class RuleGenerator(BaseEstimator):
    # parameters of the model
    approximation: Approximation = LowerApproximation()
    inclusion_measure: InclusionMeasure = ImplicatorInclusion()
    scaler_type = MinMaxScaler
    with_reducts: bool = True
    optimise_attribute_order: bool = False
    feature_ordering: FeatureOrdering = QuickReduct()
    optimise_slopes: bool = False
    slope_options: list[float] = field(default_factory=lambda: [0.001, 0.01, 0.1, 0.5, 1])
    covering_threshold: float = 1e-6
    inclusion_threshold: float = 1 - 1e-4

    # instance fields that are used during the running of the algorithm
    X = None
    y = None
    types = None
    n_samples = None
    n_attributes = None
    n_classes = None
    scaler = None
    X_scaled = None
    rel_matrix_x = None
    rel_matrix_y = None
    positive_region = None
    reducts = None
    slopes = None
    selected_indexes = None
    n_rules = None

    def __optimise_feature_order(self, X, y, t) -> np.ndarray[int]:  # todo implement
        return self.feature_ordering.order_features(X, y, t)

    def __get_reducts(self, X):
        reducts = []
        slopes = []
        if self.optimise_attribute_order:
            ordered_attributes = self.__optimise_feature_order(X, self.y, self.types)
        else:
            ordered_attributes = np.arange(self.n_attributes)

        for obj in range(self.n_samples):
            decision_set = self.rel_matrix_y[obj]
            selected_types = np.full(self.n_attributes, RelationTypes.INDISCERNIBLE, dtype=RelationTypes)
            selected_slopes = np.ones(self.n_attributes, dtype=float)
            if self.with_reducts:
                for attribute in ordered_attributes:
                    temp_types = selected_types

                    # we do UNUSED separately
                    temp_types[attribute] = RelationTypes.UNUSED
                    new_granule = fo.lukasiewicz_t_norm(
                        triangular_relation(X, X[obj], selected_slopes, temp_types),
                        self.positive_region[obj]
                    )
                    if self.inclusion_measure.inclusion(new_granule, decision_set) > self.inclusion_threshold:
                        selected_types = temp_types
                        continue  # go to the next attribute

                    # if UNUSED is not enough, we check the other types
                    found = False
                    for temp_type in [RelationTypes.DOMINATED, RelationTypes.DOMINANT, RelationTypes.INDISCERNIBLE]:
                        if found:
                            break
                        temp_types[attribute] = temp_type
                        if self.optimise_slopes:
                            temp_slopes = selected_slopes
                            for slope in self.slope_options:
                                if found:
                                    break
                                temp_slopes[attribute] = slope
                                new_granule = fo.lukasiewicz_t_norm(
                                    triangular_relation(X, X[obj], temp_slopes, temp_types),
                                    self.positive_region[obj]
                                )
                                if (self.inclusion_measure.inclusion(new_granule, decision_set)
                                        > self.inclusion_threshold):
                                    selected_types = temp_types
                                    selected_slopes = temp_slopes
                                    found = True
                        else:  # we only need to check the default slope (i.e. 1)
                            new_granule = fo.lukasiewicz_t_norm(
                                triangular_relation(X, X[obj], selected_slopes, temp_types),
                                self.positive_region[obj]
                            )
                            if (self.inclusion_measure.inclusion(new_granule, decision_set)
                                    > self.inclusion_threshold):
                                selected_types = temp_types
                                found = True

            reducts.append(selected_types)
            slopes.append(selected_slopes)
        return np.array(reducts), np.array(slopes)

    def __get_reducts_old(self, X):
        reducts = []
        current_attributes = np.random.permutation(np.arange(self.n_attributes))
        for i in range(self.n_samples):
            decision_set = self.rel_matrix_y[i]
            selected_types = 2 * np.ones(self.n_attributes, dtype=int)
            if self.with_reducts:
                for elem in current_attributes:
                    tmp_types = selected_types
                    tmp_types[elem] = 0
                    new_granule = fo.lukasiewicz_t_norm(fo.general_triangular_relation(X, X[i], self.slopes, tmp_types),
                                                        self.positive_region[i])
                    if self.inclusion_measure.inclusion(new_granule, decision_set) > self.inclusion_threshold:
                        selected_types = tmp_types
                        continue
                    tmp_types[elem] = 1
                    new_granule = fo.lukasiewicz_t_norm(fo.general_triangular_relation(X, X[i], self.slopes, tmp_types),
                                                        self.positive_region[i])
                    if self.inclusion_measure.inclusion(new_granule, decision_set) > self.inclusion_threshold:
                        selected_types = tmp_types
                        continue
                    tmp_types[elem] = -1
                    new_granule = fo.lukasiewicz_t_norm(fo.general_triangular_relation(X, X[i], self.slopes, tmp_types),
                                                        self.positive_region[i])
                    if self.inclusion_measure.inclusion(new_granule, decision_set) > self.inclusion_threshold:
                        selected_types = tmp_types
                        continue
                    tmp_types[elem] = 2
            reducts.append(selected_types)
        return np.array(reducts)

    def __optimisation_procedure(self, full_dis_covering):
        dis_model = gb.Model("discrete_rule_induction")
        # print(full_dis_covering)
        rules = []
        for r in range(self.n_samples):
            rules.append(dis_model.addVar(vtype=gb.GRB.BINARY, obj=1))
        dis_model.modelSense = gb.GRB.MINIMIZE

        for i in range(self.n_samples):
            exp = gb.quicksum([full_dis_covering[i][j] * rules[j] for j in range(self.n_samples)])
            dis_model.addConstr(exp >= 1) # , name='Coverage requirement'

        dis_model.setParam("OutputFlag", 0)
        dis_model.optimize()

        selected = []
        for i in range(self.n_samples):
            # print(rules[i])
            if rules[i].x > 0.99:
                selected.append(i)

        return np.array(selected)

    def _init_fit(self, X, y, t):
        self.X = np.atleast_2d(X)
        self.y = y
        self.types = t
        self.n_samples = self.X.shape[0]
        self.n_attributes = self.X.shape[1]
        self.n_classes = len(np.unique(y))
        self.scaler = self.scaler_type()
        self.scaler.fit(self.X)
        self.X_scaled = self.scaler.transform(self.X)
        self.rel_matrix_x = fo.triangular_similarity(self.X_scaled, self.X_scaled)
        self.rel_matrix_y = fo.discernibility_matrix(self.y, self.y)

    def fit(self, X, y, types):  # todo remove types or do something with it
        self._init_fit(X, y, types)

        self.positive_region = self.approximation.get_approximation(self.X_scaled, self.y)
        # self.reducts = self.__get_reducts_old(self.X_scaled)
        self.reducts, self.slopes = self.__get_reducts(self.X_scaled)

        covering = np.zeros((self.n_samples, self.n_samples))
        for i in range(self.n_samples):
            current_covering = fo.lukasiewicz_t_norm(
                # fo.general_triangular_relation(self.X_scaled, self.X_scaled[i], self.theta, self.reducts[i]),
                triangular_relation(
                    self.X_scaled,
                    self.X_scaled[i],
                    self.slopes[i],
                    self.reducts[i]
                ),
                self.positive_region[i]
            )
            if len(current_covering.shape) > 1:
                current_covering = current_covering.T
            covering[i, :] = (1 * (current_covering > self.covering_threshold))

        self.selected_indexes = self.__optimisation_procedure(covering.T)
        self.n_rules = np.size(self.selected_indexes)

    def predict_proba(self, X: np.ndarray, normalized: bool = True) -> np.ndarray:
        # rescale X
        X_test = self.scaler.transform(X)
        # calculate credibility of each rule
        credibility_predictions = []
        for i in range(self.n_classes):
            credibility_predictions.append([])
        for i in self.selected_indexes:
            credibility_predictions[self.y[i]].append(
                fo.lukasiewicz_t_norm(
                    # fo.general_triangular_relation(X_test, self.X_scaled[i], self.theta, self.reducts[i]),
                    triangular_relation(
                        X_test,
                        self.X_scaled[i],
                        self.slopes[i],
                        self.reducts[i]
                    ),
                    self.positive_region[i]
                ).reshape((len(X_test)))
            )

        # sum credibility for each class
        cumulative_credibility = np.zeros((X.shape[0], self.n_classes))
        for i in range(self.n_classes):
            if len(credibility_predictions[i]) != 0:  # easy fix for 0 size array problem!
                cumulative_credibility[:, i] = np.max(np.array(credibility_predictions[i]), 0)
        if normalized:
            cumulative_credibility = normalize(cumulative_credibility, norm='l1')

        return cumulative_credibility

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X, normalized=False), 1)

    def extract_rules(self):  # todo fix
        holding_points = self.X_scaled[self.selected_indexes]
        holding_reducts = self.reducts[self.selected_indexes]
        credibility = self.positive_region[self.selected_indexes]
        decisions = self.y[self.selected_indexes]
        if self.optimise_slopes:
            print("Extracting rules not yet supported for multiple slopes, just using default slope value (i.e., 1).")
            # implementation was:
            # left_bounds_t = holding_points - np.tile(self.slopes * credibility, (self.n_attributes, 1)).T
            # right_bounds_t = holding_points + np.tile(self.slopes * credibility, (self.n_attributes, 1)).T
            # I just removed the slopes
        left_bounds_t = holding_points - np.tile(credibility, (self.n_attributes, 1)).T
        right_bounds_t = holding_points + np.tile(credibility, (self.n_attributes, 1)).T
        left_bounds = self.scaler.inverse_transform(left_bounds_t)
        right_bounds = self.scaler.inverse_transform(right_bounds_t)
        self.rules = []
        for i in range(self.n_rules):
            rule = Rule(self.n_attributes)
            for j in range(self.n_attributes):
                if holding_reducts[i][j] is RelationTypes.INDISCERNIBLE:
                    rule.add_condition(j, 'i', left_bounds[i][j], right_bounds[i][j])
                elif holding_reducts[i][j] == RelationTypes.DOMINATED:
                    rule.add_condition(j, 'l', left_bounds[i][j], right_bounds[i][j])
                elif holding_reducts[i][j] == RelationTypes.DOMINANT:
                    rule.add_condition(j, 'r', left_bounds[i][j], right_bounds[i][j])
            rule.add_decision(decisions[i])
            self.rules.append(rule)

    def plot_rules2d(self, indeces):
        i = indeces[0]
        j = indeces[1]
        fig, ax = plt.subplots(1)
        colors = np.random.rand(self.n_classes, 3)
        ax.scatter(self.X[:, i], self.X[:, j], c=colors[self.y])
        ylims = ax.get_ylim()
        xlims = ax.get_xlim()
        leftylim = ylims[0]
        rightylim = ylims[1]
        leftxlim = xlims[0]
        rightxlim = xlims[1]

        for k in range(self.n_rules):
            rule = self.rules[k]
            if rule.left_bounds[i] < inf_val:
                rulexleft = rule.left_bounds[0]
            else:
                rulexleft = leftxlim
            if rule.right_bounds[i] < inf_val:
                rulexright = rule.right_bounds[0]
            else:
                rulexright = rightxlim
            if rule.left_bounds[j] < inf_val:
                ruleyleft = rule.left_bounds[1]
            else:
                ruleyleft = leftylim
            if rule.right_bounds[j] < inf_val:
                ruleyright = rule.right_bounds[1]
            else:
                ruleyright = rightylim

            col = colors[self.y[self.selected_indexes[k]]]
            rect = patches.Rectangle((rulexleft, ruleyleft),
                                     rulexright - rulexleft,
                                     ruleyright - ruleyleft,
                                     linewidth=1, edgecolor=col, facecolor='none')

            ax.add_patch(rect)
        return ax

    def get_info(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return f"@non-overlap-rules-base\n@approximation: {self.approximation}\n@scaler: {self.scaler_type()}"

    def get_rules_as_string(self) -> list[str]:
        self.extract_rules()
        return [f"{rule.condition_json}, class: {rule.decision}" for rule in self.rules]
