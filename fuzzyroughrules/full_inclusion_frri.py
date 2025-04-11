from fuzzyroughrules.approximations import LowerApproximation
from sklearn.preprocessing import normalize, MinMaxScaler
import numpy as np
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted, check_array, check_X_y

from fuzzyroughrules.feature_preprocessors import QuickReduct
from fuzzyroughrules.rule_induction_base import RuleGenerator, Rule, RelationTypes
import fuzzyroughrules.fuzzy_operators as fo
from fuzzyroughrules.operators import triangular_relation, ImplicatorInclusion


# TODO IMPLEMENT

class GranularRuleGenerator(RuleGenerator):
    """
    This version uses the full inclusion in the granular approximation during the rule shortening step.
    """

    def _get_reducts(self, X, y):
        reducts = []
        slopes = []
        if self.optimise_attribute_order:
            ordered_attributes = self._optimise_feature_order(X, y, self.types_)
        else:
            ordered_attributes = np.arange(self.n_features_in_)

        for obj in range(self.n_samples_):
            gran_approx_of_obj_class = self.gran_approx_[y[obj]]
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
                    if self.inclusion_measure_.inclusion(new_granule, gran_approx_of_obj_class) > self._get_inclusion_threshold(X[obj], y[obj]):
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
                                if (self.inclusion_measure_.inclusion(new_granule, gran_approx_of_obj_class)
                                        > self._get_inclusion_threshold(X[obj], y[obj])):
                                    new_types = temp_types
                                    new_slopes = temp_slopes
                                    found = True
                        else:  # we only need to check the default slope (i.e. 1)
                            new_granule = fo.lukasiewicz_t_norm(
                                triangular_relation(X, X[obj], new_slopes, temp_types),
                                self.positive_region_[obj]
                            )
                            if (self.inclusion_measure_.inclusion(new_granule, gran_approx_of_obj_class)
                                    > self._get_inclusion_threshold(X[obj], y[obj])):
                                new_types = temp_types
                                found = True

            reducts.append(new_types)
            slopes.append(new_slopes)
        return np.array(reducts), np.array(slopes)

    def __calculate_full_gran_approx(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Recalculates the labels of the objects in the training set X according to the result of the granular
        approximation process. This function must run after self.positive_region_ has been calculated.
        # todo add check for this final sentence

        :param X: preprocessed dataset, but before reducts
        :param y: original labels
        :return: (new labels, new positive region (approximation))
        """
        # separating out the decision classes
        separated_classes = [[] for _ in range(self.n_classes_)]  # contains obj( = index) separated by class
        for obj in range(self.n_samples_):
            separated_classes[y[obj]].append(obj)

        gran_approx = [[] for _ in range(self.n_classes_)]

        for obj in range(self.n_samples_):
            for label in range(self.n_classes_):
                if label == y[obj]:  # first case: use the solution of the optimisation problem
                    gran_approx[label].append(self.positive_region_[obj] )
                else:  # second case: compare to the objects in class label
                    gran_approx[label].append(max(
                        fo.lukasiewicz_t_norm(
                            triangular_relation(X[separated_classes[label]], X[obj]),
                            self.positive_region_[separated_classes[label]]
                        )
                    ))

        return np.array(gran_approx)

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

        # calculate positive region and perform reducts
        self.positive_region_ = np.array(self.approximation_.get_approximation(X, y))
        self.gran_approx_ = self.__calculate_full_gran_approx(X, y) # volgorde hier maakt uit! dit moet boven wat hieronder staat
        if self.apply_relabelling:
            y, self.positive_region_ = self._relabel(X, y)
        self.rel_matrix_y_ = fo.discernibility_matrix(y, y)
        self.reducts_, self.slopes_ = self._get_reducts(X, y)

        rule_candidates = range(self.n_samples_)
        if self.discard_uncertain_objects:
            rule_candidates = []
            for i in range(self.n_samples_):
                if self.positive_region_[i] >= self.certainty_threshold:
                    rule_candidates.append(i)

        if self.print_nr_of_rule_candidates:
            print(f"{len(rule_candidates)} candidates out of {self.n_samples_} possible objects")

        covering = np.zeros((len(rule_candidates), self.n_samples_))

        for index, obj in enumerate(rule_candidates):
            current_covering = fo.lukasiewicz_t_norm(
                triangular_relation(
                    X,
                    X[obj],
                    self.slopes_[obj],
                    self.reducts_[obj]
                ),
                self.positive_region_[obj]
            )
            if len(current_covering.shape) > 1:
                current_covering = current_covering.T
            covering[index, :] = (1 * (current_covering > self.covering_threshold))

        uncovered_objects = []
        if np.any(np.sum(covering, axis=0) == 0):
            uncovered_objects = np.flatnonzero(np.sum(covering, axis=0) == 0)
            if self.print_nr_of_rule_candidates:
                print(f"{np.count_nonzero(np.sum(covering, axis=0) == 0)} objects remain uncovered.")
                print("These objects will be removed from the covering matrix and can be added "
                      "as rules after the rule selection")

        selected_indexes = self._optimisation_procedure(covering.T, rule_candidates, uncovered_objects)
        self.rules_ = [Rule(
            X[i],
            self.reducts_[i],
            self.slopes_[i],
            self.positive_region_[i],
            y[i]
        ) for i in selected_indexes]
        if self.add_uncovered_objects:
            for i in uncovered_objects:
                self.rules_.append(Rule(
                    X[i],
                    self.reducts_[i],
                    self.slopes_[i],
                    self.positive_region_[i],
                    y[i]
                ))

        self.n_rules_ = np.size(selected_indexes)

        return self

    def get_info(self) -> str:
        return str(self)

    def __str__(self) -> str:
        if hasattr(self, "approximation_"):
            return f"@full_inclusion_gran_approx\n@approximation: {self.approximation_}\n@scaler: {self.scaler}"
        else:
            return f"@full_inclusion_gran_approx-NOT-FITTED"