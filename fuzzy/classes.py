from statistics import stdev

class FuzzySet:
    def __init__(self, elements=None, membership_values=None):
        # Create a FuzzySet (X, f_X) with X = elements, f_X is represented by membership_values
        if membership_values is None:
            membership_values = []
        if elements is None:
            elements = []
        self.elements = elements
        self.mfunc = dict()
        self.mvals = membership_values
        if len(membership_values) == len(elements):
            for index in range(len(membership_values)):
                self.mfunc[elements[index]] = membership_values[index]

    def add(self, x, mv):
        # Add a FuzzyElement x with membership value mv
        self.elements.append(x)
        self.mfunc[x] = mv
        self.mvals.append(mv)

    def remove(self, x):
        for index in range(len(self.elements)):
            if self.elements[index] == x:
                self.elements.pop(index)
                self.mvals.pop(index)
                break
        self.mfunc.pop(x)

    def get_mv(self, x):
        # Get membership value of element x in FuzzySet
        return self.mfunc[x]

    @staticmethod
    def subset_to_fuzzy(X, Y):
        # X: subset of Y, Y: set
        # Create a FuzzySet with elements Y and membership value mv(x) = 1 <=> x \in X, else mv(x) = 0
        elements = []
        mv = []
        for x in Y:
            elements.append(x)
            if x in X:
                mv.append(1)
            else:
                mv.append(0)
        return FuzzySet(elements, mv)

    @staticmethod
    def t_norm_min(x, y):
        return min(x, y)

    @staticmethod
    def t_norm_product(x, y):
        return x * y

    @staticmethod
    def t_norm_lukas(x, y):
        # Lukasiewicz t-norm
        return max(0, x+y-1)

    @staticmethod
    def implicator_kd(x, y):
        # Kleene-Dienes implicator
        return max(1 - x, y)

    @staticmethod
    def implicator_reichenbach(x, y):
        # Reichenbach implicator
        return 1 - x + x * y

    @staticmethod
    def implicator_lukas(x, y):
        # Lukasiewicz implicator
        return min(1, 1 - x + y)


class DecisionSystem:
    t_norm = FuzzySet.t_norm_min
    implicator = FuzzySet.implicator_kd

    def __init__(self):
        self.cases = []  # cases
        self.n_cases = 0  # amount of cases
        self.labels = []  # labels of each attribute
        self.attributes = set()  # set of attributes (excluding decision attribute)
        self.decision = None  # decision attribute
        self.decision_values = set()  # set of all possible values for the decision attribute
        self.concepts = {}  # dict of concepts, where keys are the values and self.concepts[key] is list of indices
        self.fral = False
        self.fcii = False

    def setup_variables(self):
        # Setup all different variables (e.g. the concepts, the std deviation of each attribute)
        # This is to make sure everything works accordingly
        if self.fral:
            self.labels = self.cases[0]
            self.cases = self.cases[1:]
        else:
            self.labels = [n for n in range(1, len(self.cases[0]))] + [0]

        self.n_cases = len(self.cases)

        # Create concepts
        for i, case in enumerate(self.cases):
            d = case[self.decision]
            if d in self.concepts:
                self.concepts[d].append(i)
            else:
                self.concepts[d] = [i]

        # if first_column_is_index (fcii) is True, then the first column is a case index and not an actual attribute
        # Therefore, range will be [1, decision[ (since True == 1). Otherwise, range is [0, decision[
        self.attributes = set([n for n in range(self.fcii, self.decision)])
        # Get minimum / maximum values for each attribute (None if attribute is not real-valued)
        # Additionally, convert all values in decision system to real-valued if possible
        if self.fcii:
            self.attr_max = [None]
            self.attr_min = [None]
            self.attr_real = [False]
            self.attr_stdev = [None]
        else:
            self.attr_max = []
            self.attr_min = []
            self.attr_real = []
            self.attr_stdev = []

        for case in self.cases:
            self.decision_values.add(case[self.decision])

        for a in self.attributes:
            try:
                case = self.cases[0]
                self.cases[0][a] = float(case[a])
                v = self.cases[0][a]
                min_val = v
                max_val = v
                for index, case in enumerate(self.cases[1:]):
                    self.cases[index+1][a] = float(case[a])
                    v = self.cases[index+1][a]
                    if v > max_val:
                        max_val = v
                    if v < min_val:
                        min_val = v
                # Only consider this attribute as real-valued if max_val is different from min_val
                if max_val > min_val:
                    self.attr_max.append(max_val)
                    self.attr_min.append(min_val)
                    self.attr_real.append(True)
                else:
                    self.attr_max.append(None)
                    self.attr_min.append(None)
                    self.attr_real.append(False)
                self.attr_stdev.append(stdev([case[a] for case in self.cases]))
            except ValueError:
                self.attr_max.append(None)
                self.attr_min.append(None)
                self.attr_real.append(False)
                self.attr_stdev.append(None)

    def read_from_file(self, filename, first_row_are_labels=True, first_column_is_n=True):
        # Input: filename: a filename (string), containing cases where values are separated by commas.
        # first_row_are_labels: True if the first row of the dataset are the labels of the different columns.
        # first_column_is_n: True if the first value of each case is an index number instead of an attribute).
        # Creates a decision system based on the data in the file. It is assumed that the last column is the decision
        # Read data from file
        cases = []
        f = open(filename, 'r')
        for line in f:
            if line[0] == '@':
                continue
            line = line.replace(' ', '')
            line = line.replace('\t', '')
            line = line.strip('\n')
            case = line.split(',')
            cases.append(case)
        f.close()

        # Create decision system from data
        self.cases = cases
        self.decision = len(self.cases[0]) - 1 # Get index of the last entry (which is assumed to be the decision)
        self.fral = first_row_are_labels
        self.fcii = first_column_is_n

        self.setup_variables()

    #############################################################
    # Fuzzy LEM2 Functions #
    #############################################################
    def Incl(self, A, B):
        # Returns Incl(A, B), where the inclusion measure is the implicator inclusion measure.
        return min([DecisionSystem.implicator(A[x], B[x]) for x in range(self.n_cases)])

    def R_av_case(self, a, v, case, alpha=1):
        # Calculate R_a^v(x) given a decision system S, attribute-value pair (a, v) and case.
        if self.attr_real[a]:
            return max(1 - abs(float(case[a]) - v) * alpha / self.attr_stdev[a], 0)
        else:
            return case[a] == v

    def R_av(self, a, v, x, alpha=1):
        # Calculate R_a^v(x) given a decision system S, attribute-value pair (a, v) and case with index x.
        if self.attr_real[a]:
            return max(1 - abs(self.cases[x][a] - v) * alpha / self.attr_stdev[a], 0)
        else:
            return self.cases[x][a] == v

    def R_B(self, B, x, y, alpha=1):
        # Calculate R_B(x, y) given a decision system S, a list of attributes B, and cases x and y.
        a = B.pop()
        R = self.R_av(a, self.cases[y][a], x, alpha=alpha)
        while len(B) > 0:
            a = B.pop()
            R = DecisionSystem.t_norm(R, self.R_av(a, self.cases[y][a], x, alpha=alpha))
        return R

    def calc_R_A(self, alpha=1):
        # Calculate the matrix R_A[x][y]
        R_A = []
        for x in range(self.n_cases):
            R_Ax = []
            for y in range(self.n_cases):
                R_Ax.append(self.R_B({a for a in self.attributes}, x, y, alpha=alpha))
            R_A.append(R_Ax)
        return R_A

    def block(self, F, alpha=1):
        # Calculate [F](x) = min(R_a^v(x)) for each (a, v) in F
        F_block = []
        for x in range(self.n_cases):
            m = 10
            for p in F:
                v = self.R_av(p[0], p[1], x, alpha=alpha)
                if v < m:
                    m = v
            F_block.append(m)
        return F_block

    def g_cons(self, C, alpha=1):
        # Calculate the g-consistency of concept C
        C_fuzzy = [1 if x in C else 0 for x in range(self.n_cases)]
        min_incl = 1
        for x in C:
            # Construct F such that F completely covers case x
            case = self.cases[x]
            F = {(a, case[a]) for a in self.attributes}
            F_block = self.block(F, alpha=alpha) # [F]
            incl = self.Incl(F_block, C_fuzzy)
            if incl < min_incl:
                min_incl = incl
        return min_incl

    #############################################################
    # Indiscernibility functions #
    #############################################################

    def a_indisc(self, a, x, y):
        # If attribute a is real-valued, then:
        # R_a(x, y) = 1 - |(a(x) - a(y)) / (a_max - a_min)|
        # Otherwise: R_a(x, y) = 1 <=> a(x) = a(y), else R_a(x, y) = 0
        if self.attr_real[a]:
            return 1 - abs(x[a] - y[a]) / (self.attr_max[a] - self.attr_min[a])
        else:
            return x[a] == y[a]

    def B_indisc(self, B, x, y, t=t_norm):
        # Calculate R_B(x, y) for B a subset of attributes and x, y cases (indices)
        # Where R_B(x, y) = t_{a \in B}(R_a(x, y)) with t a t-norm
        x = self.cases[x]
        y = self.cases[y]

        if len(B) == 1:
            return self.a_indisc(B[0], x, y)

        R_a = [self.a_indisc(a, x, y) for a in B]  # simplified
        # for a in B:
        #     R_a.append()

        val = DecisionSystem.t_norm(R_a[0], R_a[1])
        for i in range(2, len(B)):
            val = DecisionSystem.t_norm(val, R_a[i])
        return val

    def lower_approx(self, concept, B, t=t_norm, i=implicator):
        # Input: concept: a concept, B: a set of attributes, optionally t: t-norm and i: implicator.
        # Output: the fuzzy set which is, for y in cases,
        # (R_B \downarrow concept) = inf_{x \in cases}(i(R_B(x, y), concept)
        # = min(1 - R_B(x, y)) where min goes over all x not in concept
        la_new = FuzzySet()

        # Create an element in the fuzzy set for each case
        for x in range(self.n_cases):
            min_val = 1
            for y in range(self.n_cases):
                if concept.get_mv(y) != 1:
                    r_b = self.B_indisc(B, x, y, t)
                    if (1 - r_b) < min_val:
                        min_val = 1 - r_b
            la_new.add(x, min_val)

        return la_new
    def B_positive_region(self, B, t=t_norm, i=implicator):
        # The order of the loops is changed in order to decrease the amount of calls to get_concept and subset_to_fuzzy
        max_values = [0 for n in range(self.n_cases)]
        for d_val in self.decision_values:
            concept = self.get_concept(d_val)
            c = FuzzySet.subset_to_fuzzy(concept, [n for n in range(self.n_cases)])
            for x in range(self.n_cases):
                val = self.lower_approx(c, B, t, i).get_mv(x)
                if val > max_values[x]:
                    max_values[x] = val
        return FuzzySet([n for n in range(self.n_cases)], max_values)
    #############################################################
    # Utility #
    #############################################################
    def get_concept(self, d_val):
        concept = set()
        for index, x in enumerate(self.cases):
            if x[self.decision] == d_val:
                concept.add(index)
        return concept

    def display(self):
        # Displays the decision system as a table
        data = [self.labels] + self.cases
        col_width = max(len(str(word)) for row in data for word in row) + 2  # padding
        for row in data:
            print("".join(str(word).ljust(col_width) for word in row))
