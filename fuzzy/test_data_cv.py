from fuzzy import classes as fc
from math import sqrt
from pathlib import Path
from fuzzy.utilities import *


def R_av(a, v, c, sd, alph=1.0):
    # Calculate R_a^v(x) given a decision system S, attribute-value pair (a, v) with standard deviation sd and a case c.
    # Test if attribute is real-valued
    try:
        r = max(1 - abs(c[a] - v) * alph / sd, 0)
    except (TypeError, ValueError, ZeroDivisionError):
        r = c[a] == v
    return r


base_path = Path.cwd().parent
# Path directions
path_dataset = base_path / 'data'
# Path of dataset
path_crossval = base_path / 'output'  # Path of cross validation data

# DATA RETRIEVAL

dataset = 'crx.dat'  # Name of dataset
nosum = True
ming = 0.9

# Retrieve cases from dataset
cases = get_cases(path_dataset / dataset)  # List of cases


# Open and read the data of the cross validation file (which includes several variables used in the algorithm)

filename = dataset
if nosum:
    filename = "nosum_" + filename
if ming < 1:
    filename = "ming" + str(round(ming, 2)) + "_" + filename

f_cv = open(path_crossval / filename, 'r')
cv_data = f_cv.readlines()
f_cv.close()

# Retrieve variables from the second line
v_line = cv_data[1]
v_line = v_line.replace('#', '')
v_line = v_line.replace('\n', '')
v_line = v_line.replace(' ', '')
v_line = v_line.split(',')

N, alpha, t_norm_str, impl_str = int(v_line[0][2:]), float(v_line[1][2:]), v_line[2][7:], v_line[3][11:]

if t_norm_str == "minimum":
    t_norm = fc.FuzzySet.t_norm_min
elif t_norm_str == "product":
    t_norm = fc.FuzzySet.t_norm_product
else:  # t_norm_str == "lukas"
    t_norm = fc.FuzzySet.t_norm_lukas

if impl_str == "kd":
    implicator = fc.FuzzySet.implicator_kd
elif impl_str == "reichenbach":
    implicator = fc.FuzzySet.implicator_reichenbach
else:  # impl_str == "lukas"
    implicator = fc.FuzzySet.implicator_lukas

# Retrieve the randomised list of indices
lst = cv_data[2]
lst = lst.strip('\n')
lst = lst.strip('[]')
lst = lst.replace(' ', '')
lst = lst.split(',')
lst = [int(item) for item in lst]

# Retrieve the different concepts
concept_keys = cv_data[3]
concept_keys = concept_keys.strip('\n')
concept_keys = concept_keys.strip('[]')
concept_keys = concept_keys.replace(' ', '')
concept_keys = concept_keys.split(',')
concept_keys = [ck.strip("'") for ck in concept_keys]

# Divide data into N parts according to the randomised list
n = len(cases)
cases_per_part = n // N  # Amount of cases each part of the data will have for cross-validation
parts = [lst[i * cases_per_part:(i + 1) * cases_per_part] for i in range(N)]  # Divide lst into N parts
parts[N - 1] += lst[N * cases_per_part:n]  # If N does not divide n, add remaining cases to last part

# TESTING PHASE

# Start test
cv_data_index = 5  # Keep track of the current position within the cross-validation data
accuracies_cr = []  # List of accuracy of the certain rules, one for each of the N parts
accuracies_pr = []  # List of accuracy of the possible rules, one for each of the N parts
for i in range(5):
    # Get the test-set. The training set was only needed during Cross-Validation to get the rules (stored in f_cv)
    test_part = parts[i]  # The part used for testing (indices)
    test_data = [cases[index] for index in test_part]  # The testing dataset (cases)
    # Read the certain rules from the training data
    certain_rules = {key: [] for key in concept_keys}  # Dictionary of certain rules for each concept
    # Retrieve the standard deviation of the attributes
    st_dev = cv_data[cv_data_index]
    st_dev = st_dev.strip('\n')
    st_dev = st_dev.strip('[]')
    st_dev = st_dev.replace(' ', '')
    st_dev = st_dev.split(',')
    st_dev = [convert_str(item) for item in st_dev]
    cv_data_index += 2

    while True:
        line = cv_data[cv_data_index]
        if line[:3] == "###":  # Line is equal to "### Possible rules"
            cv_data_index += 1
            break
        elif line[0] == "#":  # Line denotes new concept starts here
            line = line.split(',')[0]
            concept = line[11:]
            cv_data_index += 1
            continue

        # Line is a new rule
        line = line.replace('\n', '')
        line = line.replace("'", '')
        line = line.strip('{}')
        rule = get_pairs_from_string(line)
        certain_rules[concept].append(rule)
        cv_data_index += 1

    # Read the possible rules from the training data
    possible_rules = {key: [] for key in concept_keys}  # Dictionary of possible rules for each concept
    while cv_data_index < len(cv_data):
        line = cv_data[cv_data_index]
        if line[:3] == "###":  # Line is equal to "### Batch i + 1"
            cv_data_index += 1
            break
        elif line[0] == "#":  # Line denotes new concept starts here
            line = line.split(',')[0]
            concept = line[11:]
            cv_data_index += 1
            continue

        # Line is a new rule
        line = line.replace('\n', '')
        line = line.replace("'", '')
        line = line.strip('{}')
        rule = get_pairs_from_string(line)
        possible_rules[concept].append(rule)
        cv_data_index += 1

    # Test the accuracy of the induced rules based on the test data
    cr_accuracy = [0 for i in range(len(test_data))]  # accuracy of the certain rules
    pr_accuracy = [0 for i in range(len(test_data))]  # accuracy of the possible rules

    counter_cr = 0
    counter_pr = 0
    for case in test_data:
        # Check, according to the induced rules, which concept this case belongs to most
        max_concept_cr = concept_keys[0]
        max_concept_pr = concept_keys[0]
        max_concept_match_cr = 0
        max_concept_match_pr = 0
        for key in concept_keys:
            max_rule_match_cr = 0
            max_rule_match_pr = 0
            # Go over each certain rule for this concept to see how well it matches with case
            for F in certain_rules[key]:
                # F = {(a_1, v_1), ..., (a_k, v_k)};
                T = 1  # T(R_a1v1(case), ..., R_akvk(case)) with T the t-norm
                for pair in F:
                    T = t_norm(T, R_av(pair[0], pair[1], case, st_dev[pair[0]], alph=alpha))  # Update t-norm
                if T > max_rule_match_cr:
                    max_rule_match_cr = T
            if max_rule_match_cr > max_concept_match_cr:
                max_concept_match_cr = max_rule_match_cr
                max_concept_cr = key

            # Go over each possible rule for this concept to see how well it matches with case
            for F in possible_rules[key]:
                # F = {(a_1, v_1), ..., (a_k, v_k)};
                T = 1  # T(R_a1v1(case), ..., R_akvk(case)) with T the t-norm
                for pair in F:
                    T = t_norm(T, R_av(pair[0], pair[1], case, st_dev[pair[0]], alph=alpha))  # Update t-norm
                if T > max_rule_match_pr:
                    max_rule_match_pr = T
            if max_rule_match_pr > max_concept_match_pr:
                max_concept_match_pr = max_rule_match_pr
                max_concept_pr = key

        if max_concept_cr == case[-1]:
            counter_cr += 1
        # else: # todo look at this maybe for else? NO, there is no break
        if max_concept_pr == case[-1]:
            counter_pr += 1
    print(
        f"cr - Correct: {counter_cr}, incorrect: {len(test_data) - counter_cr} for a total accuracy " +
        f"of {round(counter_cr / len(test_data) * 100, 2)}%")
    accuracies_cr.append(round(counter_cr / len(test_data), 4))
    print(
        f"pr - Correct: {counter_pr}, incorrect: {len(test_data) - counter_pr} for a total accuracy " +
        f"of {round(counter_pr / len(test_data) * 100, 2)}%")
    accuracies_pr.append(round(counter_pr / len(test_data), 4))

avg_cr = round(sum(accuracies_cr) / N, 4)
avg_pr = round(sum(accuracies_pr) / N, 4)

# Calculate standard deviations
st_dev_cr = 0
for acc in accuracies_cr:
    st_dev_cr += (avg_cr - acc) ** 2
st_dev_cr = sqrt(st_dev_cr)
st_dev_pr = 0
for acc in accuracies_pr:
    st_dev_pr += (avg_pr - acc) ** 2
st_dev_pr = sqrt(st_dev_pr)

print(f"Average accuracy cr: {avg_cr}; std. deviation: {st_dev_cr}. -- all accuracies: {accuracies_cr}")
print(f"Average accuracy pr: {avg_pr}; std. deviation: {st_dev_pr}. -- all accuracies: {accuracies_pr}")
