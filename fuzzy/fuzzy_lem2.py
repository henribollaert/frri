from fuzzy.classes import FuzzySet, DecisionSystem

def R_av(S, a, v, x):
    # Calculate R_a^v(x) given a decision system S, attribute-value pair (a, v) and case x.
    alpha = 2
    if S.attr_real[a]:
        return max(1 - abs(S.cases[x][a] - v)*alpha / S.attr_stdev[a], 0)
    else:
        return S.cases[x][a] == v

def R_B(S, B, x, y):
    # Calculate R_B(x, y) given a decision system S, a list of attributes B, and cases x and y.
    R = 1
    for a in B:
        R = DecisionSystem.t_norm(R, R_av(S, a, S.cases[y][a], x))
    return R

def B_pos(S, B):
    values = []
    for y in range(S.n_cases):
        mval = 0
        for key in S.concepts:
            C = FuzzySet.subset_to_fuzzy(S.concepts[key], [n for n in range(S.n_cases)])
            # min_{x \in X}{max(1 - R_B(x, y), C(x))}, where R_B(x, y) = min(R_a(x, y))
            val = min([max(1 - min([R_av(S, a, S.cases[y][a], x) for a in B]), C.get_mv(x)) for x in range(S.n_cases)])
            if val > mval:
                mval = val
        values.append(mval)
    return values

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

def pair2index(pairs, pair):
    for i in range(len(pairs)):
        if pairs[i] == pair:
            return i
    return "error"

def get_relevant_pairs(S, G, F_block, pairs, attributes_used):
    # Get all pairs t such that [t] \cap G \neq \emptyset and, supposing t = (a, v), a not in attributes_used
    # Each element of the output is an index where pairs[index] is the pair.
    relevant_pairs = set()
    A = S.attributes - attributes_used # Set of all remaining attributes
    for x in G:
        if F_block[x] < 1:
            continue
        for a in A:
            # The pair is (a, x[a]) - we transform this to an index and add it to theset of relevant pairs
            relevant_pairs.add(pair2index(pairs, (a, S.cases[x][a])))
    return relevant_pairs

def FuzzyLEM2(S, C, g):
    # Input: S = (X, A U {d}) a decision system, C a concept of S (set of integers), and g in [0, 1] a grade
    # Output: local_cover, a set of fuzzy minimal complexes of grade g of C, completely covering C
    local_cover = []
    L_block = [0 for x in range(S.n_cases)] # The grade of covering of each case in S (initially 0)
    G = C # The set of cases that still need covering
    C_fuzzy = [1 if x in C else 0 for x in range(S.n_cases)] # C_fuzzy[x] = 1 <=> x in C, otherwise C_fuzzy[x] = 0
    G_fuzzy = [1 if x in C else 0 for x in range(S.n_cases)]
    weights = [1 if x in C else -1 for x in range(S.n_cases)] # Similar as above, used when removing p from F
    # Calculate all values of R_a^v(x) for each t = (a, v) from cases in C and for each x in X
    pairs = [] # List of all pairs of cases in
    R = [] # R[t][x] = R_a^v(x), with t in [0, len(pairs)[ such that pairs[t] = (a, v)
    i = 0
    for a in S.attributes:
        for case in C:
            v = S.cases[case][a]
            if (a, v) in pairs: # Check if this specific attribute-pair combination has already appeared
                continue
            pairs.append((a, v))
            R.append([])
            for x in range(S.n_cases):
                # Calculate
                R[i].append(R_av(S, a, v, x))
            i += 1

    # Start the actual algorithm
    while len(G) > 0: # While there are still cases left to be covered
        F = set() # Fuzzy minimal complex being generated
        F_block = [1 for n in range(S.n_cases)] # [F], initially 1 for each case
        Incl = S.Incl(F_block, C_fuzzy)
        attributes_used = set() # F cannot use the same attribute twice

        while Incl < g: # while Incl([F], C) < g
            # Get all relevant pairs
            relevant_pairs = get_relevant_pairs(S, G, F_block, pairs, attributes_used)

            # From these, get the optimal pair. That is: the pair t for which sum_{x in X}(A_t(x)) is maximal
            # Calculate A_t(x) for each pair t and each case x, keeping track of optimal pair
            max_pair = None
            max_A_t = 0
            for p in relevant_pairs:
                A_t = 0
                for x in range(S.n_cases):
                    if x in G:
                        # A_t += 1 - max(F_block[x] - max(R[p][x], L_block[x]), 0)
                        A_t += 1 - max(F_block[x] - R[p][x], L_block[x])
                    elif x not in C and F_block[x] > 0: # x not in C, and [F] not equal to 0
                        A_t += max(F_block[x] - R[p][x], 0) / F_block[x]
                    # else: A_t += 0, but this would be redundant :)

                if A_t > max_A_t:
                    max_A_t = A_t
                    max_pair = p

            F.add(max_pair) # Add max_pair to F
            test1 = [F_block[x] for x in range(S.n_cases)]
            test2 = [R[max_pair][x] for x in range(S.n_cases)]
            F_block = [min(F_block[x], R[max_pair][x]) for x in range(S.n_cases)] # Update [F]
            Incl = S.Incl(F_block, C_fuzzy) # Update value for Incl
            attributes_used.add(pairs[max_pair][0]) # Update used attributes for this F
        F = {pairs[p] for p in F} # Redefine F such that it now contains actual pairs instead of indices
        # Check each pair in F for whether or not it can be removed
        F_new = {p for p in F} # Copy of F for the removal process
        if len(F) > 1:
            for p in F:
                # Remove p from F if:
                # 1) sum_{x in C}([F \ {p}](x) - [F](x)) - sum_{x in X\C}([F \ {p}](x) - [F](x))
                # 2) Incl([F \ {p}]) >= g
                F_new_b = block(S, F - {p})
                # s = sum([weights[x]*(F_new_b[x] - F_block[x]) for x in range(S.n_cases)])
                s = 0
                if s >= 0 and S.Incl(F_new_b, C_fuzzy) >= g:
                    F_new -= {p}
                    F_block = block(S, F_new)
        F = {p for p in F_new}

        local_cover.append(F)
        L_block = [max(L_block[x], F_block[x]) for x in range(S.n_cases)]
        G = {x for x in G if L_block[x] < 1}
        G_fuzzy = [1 if x in G else 0 for x in range(S.n_cases)]

    # Remove possible unnecessary minimal concepts from L
    # First calculate [F] for each F in L
    F_blocks = []
    for F in local_cover:
        # Calculate F_block
        F_blocks.append(block(S, F))

    Incl_L = S.Incl(L_block, C_fuzzy) # Calculate Incl([L], C)
    # Try to remove minimal complexes from the local cover (if it has at least 2 minimal complexes)
    if len(local_cover) > 1:
        # Instead of adding/removing minimal concepts themselves, we will work with indices instead
        L_indices = [n for n in range(len(local_cover))]
        for F_n in range(len(local_cover)): # F_n stands the index of F, for each F in L
            # Remove F from L and see if [L \ F] still
            r = [n for n in L_indices if n != F_n] # L' = L \ F
            L_new_block = [max([F_blocks[F][n] for F in r]) for n in range(S.n_cases)]
            # [L']
            m = min([L_new_block[x] for x in C]) # The minimal value [L'](x) for x in C
            if m == 1 and S.Incl(L_new_block, C_fuzzy) >= Incl_L:
                L_indices.remove(F_n)
        local_cover = [local_cover[n] for n in L_indices]
    return local_cover
