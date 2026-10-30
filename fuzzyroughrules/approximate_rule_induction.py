"""
This file contains the implementation of approximate rule induction, where the exact (gurobi) solver is
replaced by an approximate vertex cover solver.
"""
from dataclasses import dataclass
from typing import override

from fuzzyroughrules.rule_induction_base import RuleGenerator


@dataclass
class ApproximateRuleInduction(RuleGenerator):

    @override
    def _optimisation_procedure(
            self,
            full_dis_covering,
            candidates = None,
            uncovered = None,
    ):
        ...