from quickrules.quickrules import *
from quickrules.weights import Weights


class OWAQuickRules(QuickRules):
    """
    Variant of QuickRules that uses the OWA-based fuzzy rough lower approximation.
    """
    def __init__(
            self,
            t_norm: TNorm,
            implicator: Implicator,
            relation_factory: RelationFactory,
            weight_function: Weights,
            prune: bool = False,
            combo: bool = True  # signifies the prediction component
    ):
        super().__init__(t_norm, implicator, relation_factory, prune, combo)
        self.weight_function = weight_function
        self.weights: np.ndarray = Optional[None]

    def get_info(self) -> str:
        return "@owa-quickrules\n" + super().get_info()

    def _init_fit(self, x: np.ndarray, y: np.ndarray):
        super()._init_fit(x, y)
        self.weights = self.weight_function(self.X.shape[0])

    def _calculate_single_pos_membership(self,
                                         sample: np.ndarray,
                                         label: LabelType,
                                         attributes: list[bool] = None) -> float:
        """
        Calculates the membership to the OWA-based fuzzy-rough lower approximation.
        :param sample:
        :param label:
        :param attributes:
        :return:
        """
        if attributes is None:
            attributes = [True] * self.nr_of_attributes
        memberships = [self.implicator(
            self.relation(sample,
                          other,
                          attributes),
            int(label == other_label)) for other, other_label in zip(self.X, self.y)]
        memberships.sort()  # memberships are sorted ascending -> we want to emulate "min"
        weighted_memberships = [w * m for w, m in zip(self.weights, memberships)]
        return sum(weighted_memberships)
