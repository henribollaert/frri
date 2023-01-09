from dataclasses import dataclass


@dataclass
class Rule:  # todo figure out types
    antecedent: ...  # set of conditional attributes in the rule's antecedent
    generating_set: ...  # fuzzy tolerance class of the object that generated the rule
    decision: ...  # the decision class/consequent of the rule


class QuickRules:

    def __init__(self):
        self.rules = None
        self.covered = None

    def quick_rules(self):
        conditions, self.rules, self.covered = [], [], []

    def check(self, rule_to_check: Rule):
        add = True
        for rule in self.rules:
            if rule_to_check.generating_set in rule.generating_set:
                add = False
                break
            elif rule.generating_set in rule_to_check:
                del rule
        if add:
            self.rules.append(rule_to_check)
            self.covered.append(rule_to_check.generating_set)

