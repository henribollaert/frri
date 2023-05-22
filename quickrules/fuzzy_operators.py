"""
File containing fuzzy operators
"""


class MinTNorm:
    def __call__(self, a, b):
        return min(a, b)

    def __repr__(self):
        return "Minimum t-norm"


class LukasiewiczTNorm:
    def __call__(self, a, b):
        return max(0, a + b - 1)

    def __repr__(self):
        return "Lukasiewicz t-norm"


class KleeneDienesImplicator:
    def __call__(self, a, b):
        return max(1 - a, b)

    def __repr__(self):
        return "Kleene-Dienes implicator"
