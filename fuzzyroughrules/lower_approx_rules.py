import numpy as np
from sklearn.base import BaseEstimator
import gurobipy as gb
import matplotlib.pyplot as plt
import matplotlib.patches as patches
np.set_printoptions(formatter={'float': lambda x: "{0:0.5f}".format(x)})
from sklearn.preprocessing import MinMaxScaler
inf_val = float('Inf')
import fuzzyroughrules.fuzzy_operators as fo
import importlib
importlib.reload(fo)


class Rule:
    def __init__(self, size):
        self.size = size
        self.left_bounds = inf_val*np.ones(self.size)
        self.right_bounds = inf_val*np.ones(self.size)
        self.condition_json = []
        self.decision = 0

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


class RuleGenerator(BaseEstimator):

    def __init__(
            self,
            scaler_type=MinMaxScaler,
            with_reducts=True,
            tol=1e-4,
            theta=1
    ):
        self.scaler_type = scaler_type
        self.with_reducts = with_reducts
        self.theta = theta
        self.tol = tol

    def __get_reducts(self, X):
        reducts = []
        current_attributes = np.random.permutation(np.arange(self.n_attributes))
        for i in range(self.n_samples):
            decision_set = self.rel_matrix_y[i]
            selected_types = 2*np.ones(self.n_attributes, dtype=int)
            if self.with_reducts:
                for elem in current_attributes:
                    tmp_types = selected_types
                    tmp_types[elem] = 0
                    new_granule = fo.lukasiewicz_t_norm(fo.general_triangular_relation(X, X[i], self.theta, tmp_types), self.positive_region[i])
                    if fo.is_subset(new_granule, decision_set, self.tol):
                        selected_types = tmp_types
                        continue
                    tmp_types[elem] = 1
                    new_granule = fo.lukasiewicz_t_norm(fo.general_triangular_relation(X, X[i], self.theta, tmp_types), self.positive_region[i])
                    if fo.is_subset(new_granule, decision_set, self.tol):
                        selected_types = tmp_types
                        continue
                    tmp_types[elem] = -1
                    new_granule = fo.lukasiewicz_t_norm(fo.general_triangular_relation(X, X[i], self.theta, tmp_types), self.positive_region[i])
                    if fo.is_subset(new_granule, decision_set, self.tol):
                        selected_types = tmp_types
                        continue
                    tmp_types[elem] = 2
            reducts.append(selected_types)
        return np.array(reducts)

    def __optimisation_procedure(self, full_dis_covering):
        dis_model = gb.Model("discrete_rule_induction")

        rules = []
        for r in range(self.n_samples):
            rules.append(dis_model.addVar(vtype=gb.GRB.BINARY, obj=1))
        dis_model.modelSense = gb.GRB.MINIMIZE

        for i in range(self.n_samples):
            exp = gb.quicksum([full_dis_covering[i][j] * rules[j] for j in range(self.n_samples)])
            dis_model.addConstr(exp >= 1)

        dis_model.setParam("OutputFlag", 0)
        dis_model.optimize()

        selected = []
        for i in range(self.n_samples):
            # print(rules[i])
            if rules[i].x > 0.99:
                selected.append(i)

        return np.array(selected)

    def fit(self, data, labels, types):  # used to be get_rules *hb
        self.X = np.atleast_2d(data)
        self.train_prediction = labels
        self.n_samples = self.X.shape[0]
        self.n_attributes = self.X.shape[1]
        self.n_classes = len(np.unique(labels))
        self.scaler = self.scaler_type()
        self.scaler.fit(self.X)
        self.X_scaled = self.scaler.transform(self.X)
        self.rel_matrix_x = fo.triangular_similarity(self.X_scaled, self.X_scaled)
        self.rel_matrix_y = fo.discernibility_matrix(self.train_prediction, self.train_prediction)
        '''
        fuzzy = []
        indexes = []
        low_apr = []
        new_low_apr = []
        self.positive_region = np.zeros(self.n_samples)
        self.low_apr_indexes = np.zeros(self.n_samples, dtype=int)

        for i in range(self.n_classes):
            label = 1 * (train_prediction == i)
            indexes.append(np.where(train_prediction == i)[0])
            fuzzy.append(label)
            current_low_app = fo.get_ind_apr3(self.X_train, fuzzy[i], theta=self.theta)
            current_low_app = fo.lukasiewicz_t_norm(current_low_app, 0.5)
            low_apr.append(current_low_app)
        low_apr = np.array(low_apr)
         self.positive_region = np.max(low_apr, 0)
        '''
        init_set = .5*np.ones(self.n_samples)
        self.positive_region = fo.get_ind_low_apr(self.rel_matrix_x, self.rel_matrix_y)  # as first param: init_set,

        self.reducts = self.__get_reducts(self.X_scaled)

        covering = np.zeros((self.n_samples, self.n_samples))
        for i in range(self.n_samples):
            current_covering = fo.lukasiewicz_t_norm(fo.general_triangular_relation(self.X_scaled, self.X_scaled[i], self.theta, self.reducts[i]), self.positive_region[i])
            if len(current_covering.shape) > 1:
                current_covering = current_covering.T
            # print(current_covering)
            covering[i, :] = (1*(current_covering > 1e-6))
        # print(covering)
        self.selected_indexes = self.__optimisation_procedure(covering.T)
        self.n_rules = np.size(self.selected_indexes)

    def predict(self, X):
        X_test = self.scaler.transform(X)
        credibility_predictions = []
        for i in range(self.n_classes):
            credibility_predictions.append([])
        for i in self.selected_indexes:
            credibility_predictions[self.train_prediction[i]].append(
                fo.lukasiewicz_t_norm(fo.general_triangular_relation(X_test, self.X_scaled[i], self.theta, self.reducts[i]), 
                self.positive_region[i])
            )
        cumulative_credibility = []
        for i in range(self.n_classes):
            cumulative_credibility.append(np.max(np.array(credibility_predictions[i]), 0))
        prediction = np.argmax(np.array(cumulative_credibility), 0)
        return prediction

    def extract_rules(self):
        holding_points = self.X_scaled[self.selected_indexes]
        holding_reducts = self.reducts[self.selected_indexes]
        credibility = self.positive_region[self.selected_indexes]
        decisions = self.train_prediction[self.selected_indexes]
        left_bounds_t = holding_points - np.tile(self.theta*credibility, (self.n_attributes, 1)).T
        right_bounds_t = holding_points + np.tile(self.theta*credibility, (self.n_attributes, 1)).T
        left_bounds = self.scaler.inverse_transform(left_bounds_t)
        right_bounds = self.scaler.inverse_transform(right_bounds_t)
        self.rules = []
        for i in range(self.n_rules):
            rule = Rule(self.n_attributes)
            for j in range(self.n_attributes):
                pair = {}
                if holding_reducts[i][j] == 2:
                    rule.add_condition(j,'i',left_bounds[i][j], right_bounds[i][j])
                elif holding_reducts[i][j] == 1:
                    rule.add_condition(j, 'l', left_bounds[i][j], right_bounds[i][j])
                elif holding_reducts[i][j] == -1:
                    rule.add_condition(j, 'r', left_bounds[i][j], right_bounds[i][j])
            rule.add_decision(decisions[i])
            self.rules.append(rule)

    def plot_rules2d(self, indeces):
        i = indeces[0]
        j = indeces[1]
        fig, ax = plt.subplots(1)
        colors = np.random.rand(self.n_classes, 3)
        ax.scatter(self.X[:, i], self.X[:, j], c=colors[self.train_prediction])
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
            
            col = colors[self.train_prediction[self.selected_indexes[k]]]
            rect = patches.Rectangle((rulexleft, ruleyleft),
                                     rulexright - rulexleft,
                                     ruleyright - ruleyleft,
                                     linewidth=1, edgecolor=col, facecolor='none')

            ax.add_patch(rect)
        return ax

    def get_info(self) -> str:
        return f"@lower-approx-rules-base\n@scaler: {self.scaler_type()}"

    def get_rules_as_string(self) -> list[str]:
        self.extract_rules()
        return [f"{rule.condition_json}, class: {rule.decision}" for rule in self.rules]

