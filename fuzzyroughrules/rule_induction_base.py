from typing import Protocol
from dataclasses import dataclass, field

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import normalize, MinMaxScaler
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
import gurobipy as gb
from fuzzyroughrules.operators import RelationTypes, triangular_relation, ImplicatorInclusion
from fuzzyroughrules.approximations import LowerApproximation
from fuzzyroughrules.feature_preprocessors import QuickReduct

np.set_printoptions(formatter={'float': lambda x: "{0:0.5f}".format(x)})

inf_val = float('Inf')
import fuzzyroughrules.fuzzy_operators as fo
import importlib

importlib.reload(fo)


class OldRule:
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


@dataclass
class Rule:
    antecedent: np.ndarray
    reducts: np.ndarray
    slopes: np.ndarray
    credibility: np.ndarray
    decision: int
    as_string: str = field(default_factory=lambda: "Not yet extracted")


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
class RuleGenerator(BaseEstimator, ClassifierMixin):
    # parameters of the model
    with_reducts: bool = True
    apply_relabelling: bool = False
    print_changes: bool = False
    optimise_attribute_order: bool = False
    optimise_slopes: bool = False
    slope_options: list[float] = None
    covering_threshold: float = 1e-6
    inclusion_threshold: float = 1 - 1e-6
    priors_influence: float = 0
    approximation: Approximation = None
    inclusion_measure: InclusionMeasure = None
    attribute_ordering: FeatureOrdering = None
    scaler = None

    def set_params(self, **params):
        if not params:
            return self

        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"RuleGenerator does not have a {key} parameter.")
        return self

    def __optimise_feature_order(self, X, y, t) -> np.ndarray[int]:  # todo implement
        return self.attribute_ordering_.order_features(X, y, t)

    def __get_inclusion_threshold(self, obj, label) -> float:
        """

        :param obj: object
        :param label: label
        :return:
        """
        if not self.priors_influence:
            return self.inclusion_threshold
        return (self.priors_[label] * self.priors_influence + 1 - self.priors_influence) * self.inclusion_threshold  # high scale

    def __get_reducts(self, X, y):
        reducts = []
        slopes = []
        if self.optimise_attribute_order:
            ordered_attributes = self.__optimise_feature_order(X, y, self.types_)
        else:
            ordered_attributes = np.arange(self.n_features_in_)

        for obj in range(self.n_samples_):
            decision_set = self.rel_matrix_y_[obj]
            new_types = np.full(self.n_features_in_, RelationTypes.INDISCERNIBLE, dtype=RelationTypes)
            new_slopes = np.ones(self.n_features_in_, dtype=float)

            if self.with_reducts:
                for attribute in ordered_attributes:
                    temp_types = new_types

                    # we do UNUSED separately because then the slopes don't matter
                    temp_types[attribute] = RelationTypes.UNUSED
                    new_granule = fo.lukasiewicz_t_norm(
                        triangular_relation(X, X[obj], new_slopes, temp_types),
                        self.positive_region_[obj]
                    )
                    if self.inclusion_measure_.inclusion(new_granule, decision_set) > self.__get_inclusion_threshold(X[obj], y[obj]):
                        new_types = temp_types
                        continue  # go to the next attribute

                    # if UNUSED is not enough, we check the other types
                    found = False
                    for temp_type in [RelationTypes.DOMINATED, RelationTypes.DOMINANT, RelationTypes.INDISCERNIBLE]:
                        if found:
                            break
                        temp_types[attribute] = temp_type
                        if self.optimise_slopes:
                            temp_slopes = new_slopes
                            for slope in self.slope_options_:
                                if found:
                                    break
                                temp_slopes[attribute] = slope
                                new_granule = fo.lukasiewicz_t_norm(
                                    triangular_relation(X, X[obj], temp_slopes, temp_types),
                                    self.positive_region_[obj]
                                )
                                if (self.inclusion_measure_.inclusion(new_granule, decision_set)
                                        > self.__get_inclusion_threshold(X[obj], y[obj])):
                                    new_types = temp_types
                                    new_slopes = temp_slopes
                                    found = True
                        else:  # we only need to check the default slope (i.e. 1)
                            new_granule = fo.lukasiewicz_t_norm(
                                triangular_relation(X, X[obj], new_slopes, temp_types),
                                self.positive_region_[obj]
                            )
                            if (self.inclusion_measure_.inclusion(new_granule, decision_set)
                                    > self.__get_inclusion_threshold(X[obj], y[obj])):
                                new_types = temp_types
                                found = True

            reducts.append(new_types)
            slopes.append(new_slopes)
        return np.array(reducts), np.array(slopes)

    def __optimisation_procedure(self, full_dis_covering):
        dis_model = gb.Model("discrete_rule_induction")
        # print(full_dis_covering)
        rules = []
        for r in range(self.n_samples_):
            rules.append(dis_model.addVar(vtype=gb.GRB.BINARY, obj=1))
        dis_model.modelSense = gb.GRB.MINIMIZE

        for i in range(self.n_samples_):
            exp = gb.quicksum([full_dis_covering[i][j] * rules[j] for j in range(self.n_samples_)])
            dis_model.addConstr(exp >= 1)  # , name='Coverage requirement'

        dis_model.setParam("OutputFlag", 0)
        dis_model.optimize()

        selected = []
        for i in range(self.n_samples_):
            if rules[i].x > 0.99:
                selected.append(i)

        return np.array(selected)

    def __relabel(self, X: np.ndarray, y: np.ndarray) -> (np.ndarray, np.ndarray):
        """
        Recalculates the labels of the objects in the training set X according to the result of the granular
        approximation process. This function must run after self.positive_region_ has been calculated.
        # todo add check for this final sentence

        :param X: preprocessed dataset, but before reducts
        :param y:
        :return: (new labels, new positive region (approximation))
        """
        # start by separating out the decision classes?
        separated_classes = [[] for _ in range(self.n_classes_)]  # contains obj( = index) separated by class
        for obj in range(self.n_samples_):
            separated_classes[y[obj]].append(obj)

        new_y = []
        new_gran_approx = []
        nr_changes = 0

        for obj in range(self.n_samples_):
            best_label = y[obj]
            best_membership = self.positive_region_[obj]  # first case -> use the solution of the optimisation problem
            for label in range(self.n_classes_):
                if label != y[obj]:  # second case: compare to the objects in class label
                    temp = max(
                        fo.lukasiewicz_t_norm(
                            triangular_relation(X[separated_classes[label]], X[obj]),
                            self.positive_region_[separated_classes[label]]
                        )
                    )
                    if temp > best_membership:
                        best_membership = temp
                        best_label = label
            if best_label != y[obj]:
                nr_changes += 1
            new_y.append(best_label)
            new_gran_approx.append(best_membership)

        if self.print_changes:
            print(f"{nr_changes} labels out of {self.n_samples_} were changed.")
            # print(new_gran_approx)

        return new_y, new_gran_approx

    def fit(self, X: np.ndarray, y: np.ndarray, types: np.ndarray = None):
        # initialise fit
        X, y = check_X_y(X, y)
        check_classification_targets(y)
        self.classes_, y, counts = np.unique(y, return_inverse=True, return_counts=True)
        self.priors_ = np.divide(counts, max(counts))  # scaled_priors
        self.n_classes_ = len(self.classes_)
        self.types_ = types
        self.n_samples_, self.n_features_in_ = X.shape

        self.approximation_ = LowerApproximation() if self.approximation is None else self.approximation
        self.inclusion_measure_ = ImplicatorInclusion() if self.inclusion_measure is None else self.inclusion_measure
        self.attribute_ordering_ = QuickReduct() if self.attribute_ordering is None else self.attribute_ordering
        self.scaler_ = MinMaxScaler() if self.scaler is None else self.scaler
        self.slope_options_ = [0.001, 0.01, 0.1, 0.5, 1] if self.slope_options is None else self.slope_options


        X = self.scaler_.fit_transform(X)
        # self.rel_matrix_x_ = fo.triangular_similarity(X, X)

        # calculate positive region and perform reducts
        self.positive_region_ = self.approximation_.get_approximation(X, y)
        if self.apply_relabelling:
            y, self.positive_region_ = self.__relabel(X, y)
        self.rel_matrix_y_ = fo.discernibility_matrix(y, y)
        self.reducts_, self.slopes_ = self.__get_reducts(X, y)

        covering = np.zeros((self.n_samples_, self.n_samples_))

        for i in range(self.n_samples_):
            current_covering = fo.lukasiewicz_t_norm(
                triangular_relation(
                    X,
                    X[i],
                    self.slopes_[i],
                    self.reducts_[i]
                ),
                self.positive_region_[i]
            )
            if len(current_covering.shape) > 1:
                current_covering = current_covering.T
            covering[i, :] = (1 * (current_covering > self.covering_threshold))

        selected_indexes = self.__optimisation_procedure(covering.T)
        self.rules_ = [Rule(
            X[i],
            self.reducts_[i],
            self.slopes_[i],
            self.positive_region_[i],
            y[i]
        ) for i in selected_indexes]

        self.n_rules_ = np.size(selected_indexes)

        return self

    def predict_proba(self, X: np.ndarray, normalized: bool = True) -> np.ndarray:
        check_is_fitted(self)
        # rescale X
        X = check_array(X)
        X_test = self.scaler_.transform(X)
        # calculate credibility of each rule
        credibility_predictions = []
        for i in range(self.n_classes_):
            credibility_predictions.append([])
        for rule in self.rules_:
            credibility_predictions[rule.decision].append(
                fo.lukasiewicz_t_norm(
                    triangular_relation(
                        X_test,
                        rule.antecedent,
                        rule.slopes,
                        rule.reducts
                    ),
                    rule.credibility
                ).reshape((len(X_test)))
            )

        # sum credibility for each class
        cumulative_credibility = np.zeros((X.shape[0], self.n_classes_))
        for i in range(self.n_classes_):
            if len(credibility_predictions[i]) != 0:  # easy fix for 0 size array problem!
                cumulative_credibility[:, i] = np.max(np.array(credibility_predictions[i]), 0)
        if normalized:
            cumulative_credibility = normalize(cumulative_credibility, norm='l1')

        return cumulative_credibility

    def predict(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self)
        return self.classes_[np.argmax(self.predict_proba(X, normalized=False), 1)]

    # def plot_rules2d(self, indeces):
    #     i = indeces[0]
    #     j = indeces[1]
    #     fig, ax = plt.subplots(1)
    #     colors = np.random.rand(self.n_features_in_, 3)
    #     ax.scatter(self.X[:, i], self.X[:, j], c=colors[self.y])
    #     ylims = ax.get_ylim()
    #     xlims = ax.get_xlim()
    #     leftylim = ylims[0]
    #     rightylim = ylims[1]
    #     leftxlim = xlims[0]
    #     rightxlim = xlims[1]
    #
    #     for k in range(self.n_rules):
    #         rule = self.rules[k]
    #         if rule.left_bounds[i] < inf_val:
    #             rulexleft = rule.left_bounds[0]
    #         else:
    #             rulexleft = leftxlim
    #         if rule.right_bounds[i] < inf_val:
    #             rulexright = rule.right_bounds[0]
    #         else:
    #             rulexright = rightxlim
    #         if rule.left_bounds[j] < inf_val:
    #             ruleyleft = rule.left_bounds[1]
    #         else:
    #             ruleyleft = leftylim
    #         if rule.right_bounds[j] < inf_val:
    #             ruleyright = rule.right_bounds[1]
    #         else:
    #             ruleyright = rightylim
    #
    #         col = colors[self.y[self.selected_indexes[k]]]
    #         rect = patches.Rectangle((rulexleft, ruleyleft),
    #                                  rulexright - rulexleft,
    #                                  ruleyright - ruleyleft,
    #                                  linewidth=1, edgecolor=col, facecolor='none')
    #
    #         ax.add_patch(rect)
    #     return ax

    def get_info(self) -> str:
        return str(self)

    def __str__(self) -> str:
        if hasattr(self, "approximation_"):
            return f"@non-overlap-rules-base\n@approximation: {self.approximation_}\n@scaler: {self.scaler}"
        else:
            return f"@non-overlap-rules-base-NOT-YET-FITTED"

    def get_rules_as_string(self) -> list[str]:
        if self.optimise_slopes:
            print("Extracting rules not yet supported for multiple slopes, just using default slope value (i.e., 1).")
            # implementation was:
            # left_bounds_t = holding_points - np.tile(self.slopes * credibility, (self.n_attributes, 1)).T
            # right_bounds_t = holding_points + np.tile(self.slopes * credibility, (self.n_attributes, 1)).T
            # I just removed the slopes
        old_rules = []
        for rule in self.rules_:
            left_bound_t = rule.antecedent - np.tile(rule.credibility, (self.n_features_in_, 1)).T
            right_bound_t = rule.antecedent + np.tile(rule.credibility, (self.n_features_in_, 1)).T
            left_bound = np.squeeze(self.scaler_.inverse_transform(left_bound_t))
            right_bound = np.squeeze(self.scaler_.inverse_transform(right_bound_t))

            created_old_rule = OldRule(self.n_features_in_)
            for j in range(self.n_features_in_):
                if rule.reducts[j] is RelationTypes.INDISCERNIBLE:
                    created_old_rule.add_condition(j, 'i', left_bound[j], right_bound[j])
                elif rule.reducts[j] == RelationTypes.DOMINATED:
                    created_old_rule.add_condition(j, 'l', left_bound[j], right_bound[j])
                elif rule.reducts[j] == RelationTypes.DOMINANT:
                    created_old_rule.add_condition(j, 'r', left_bound[j], right_bound[j])
            created_old_rule.add_decision(rule.decision)
            old_rules.append(created_old_rule)
        return [f"{rule.condition_json}, class: {rule.decision}" for rule in old_rules]
