from fuzzy.classes import *
from fuzzy import fuzzy_lem2


# todo why are R_av and such calculated multiple times?

def R_av(S, a, v, x):
    # Calculate R_a^v(x) given a decision system S, attribute-value pair (a, v) and case x.
    alpha = 2
    if S.attr_real[a]:
        return max(1 - abs(S.cases[x][a] - v) * alpha / S.attr_stdev[a], 0)
    else:
        return S.cases[x][a] == v


def R_B(S, B, x, y):
    # Calculate R_B(x, y) given a decision system S, a list of attributes B, and cases x and y.
    a = B.pop()
    R = R_av(S, a, S.cases[y][a], x)
    if len(B) == 0:
        return R
    while len(B) > 0:
        a = B.pop()
        R = DecisionSystem.t_norm(R, R_av(S, a, S.cases[y][a], x))
    return R


def calc_R_A(S):
    # Calculate the matrix R_A[x][y]
    R_A = []
    for x in range(S.n_cases):
        R_Ax = []
        for y in range(S.n_cases):
            R_Ax.append(R_B(S, {a for a in S.attributes}, x, y))
        R_A.append(R_Ax)
    return R_A


def block(S, F):
    # Calculate [F](x) = min(R_a^v(x)) for each (a, v) in F
    F_block = []
    for x in range(S.n_cases):
        m = 10
        for p in F:
            v = R_av(S, p[0], p[1], x)
            if v < m:
                m = v
        F_block.append(m)
    return F_block


def g_cons(S, C):
    C_fuzzy = [1 if x in C else 0 for x in range(S.n_cases)]
    kernel = [x for x in C]
    min_incl = 1
    for x in C:
        # Construct F such that F completely covers case x
        case = S.cases[x]
        F = {(a, case[a]) for a in S.attributes}
        F_block = block(S, F)  # [F]
        incl = S.Incl(F_block, C_fuzzy)

        if incl < 1:
            for y in kernel:
                if F_block[y] > 0:
                    kernel.remove(y)

        if incl < min_incl:
            min_incl = incl
    return min_incl


def FuzzyRoughLEM2(S, C, d_val, a=1):
    A = set([n for n in range(len(S.cases[0]) - 1)])  # set of attributes
    fuzzy_concept = [1 if x in C else 0 for x in range(S.n_cases)]
    R_A = S.calc_R_A(alpha=a)

    new_S = DecisionSystem()  # Create a new decision system
    new_S.decision = S.decision  # Set the decision value to the correct index
    new_S.fral = False  # The first row of new_S is data, not labels, hence 'first row are labels = False'
    new_S.fcii = False  # The first column of new_S is data, not indices, hence 'first column is index = False'
    new_C = []  # The new concept for new_S

    cons = S.g_cons(C, alpha=a)
    for x, case in enumerate(S.cases):
        new_case = [v for v in case]
        lapr = 1
        # Calculate the lower approximation of x
        for y in range(S.n_cases):
            val_y = DecisionSystem.implicator(R_A[x][y], fuzzy_concept[y])
            if val_y < lapr:
                lapr = val_y

        if lapr == 1:  # If min( I( R_A(x, y), C(y) ) ) = 1, i.e. if the lower approximation l_apr(x) = 1
            new_case[S.decision] = d_val  # Add this case to the concept
            new_C.append(x)
        elif x not in C:
            new_case[S.decision] = 'SPECIAL'  # Let this case be in a different concept
        new_S.cases.append(new_case)

    new_S.setup_variables()  # Setup all other variables accordingly
    g_cr = g_cons(new_S, new_C)

    certain_rules = fuzzy_lem2.FuzzyLEM2(new_S, new_C, min(g_cr, 0.9))
    # If the decision system is 1-consistent, then upper approximation = lower approximation
    if cons == 1:
        # Since possible_rules = certain_rules in this case, we can immediately return certain_rules twice
        return certain_rules, certain_rules, g_cr, g_cr

    new_S = DecisionSystem()  # Create a new decision system
    new_S.decision = S.decision  # Set the decision value to the correct index
    new_S.fral = False  # The first row of new_S is data, not labels, hence 'first row are labels = False'
    new_S.fcii = False  # The first column of new_S is data, not indices, hence 'first column is index = False'
    new_C = []  # The new concept for new_S
    for x, case in enumerate(S.cases):
        new_case = case
        max_val = 0  # This will be equal to max( T( R_A(x, y), C(y) ) )
        for y in range(S.n_cases):
            val_y = DecisionSystem.t_norm(R_A[x][y], fuzzy_concept[y])
            if val_y > max_val:
                max_val = val_y

        if max_val > 0:  # If max( T( R_A(x, y), C(y) ) ) > 0, i.e. if the upper approximation u_apr(x) > 0
            new_case[S.decision] = d_val
            new_C.append(x)
        else:
            new_case[S.decision] = 'SPECIAL'
        new_S.cases.append(new_case)
    new_S.setup_variables()  # Setup all other variables accordingly
    g_pr = g_cons(new_S, new_C)

    possible_rules = fuzzy_lem2.FuzzyLEM2(new_S, new_C, min(g_pr, 0.9))

    return certain_rules, possible_rules, g_cr, g_pr
