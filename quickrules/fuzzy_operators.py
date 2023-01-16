class MinTNorm:
    def __call__(self, a, b):
        return min(a, b)


class KleeneDienesImplicator:
    def __call__(self, a, b):
        return max(1 - a, b)
