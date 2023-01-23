from random import shuffle
from fuzzy import classes as fc
from fuzzy import fuzzy_rough_lem2
from pathlib import Path
from fuzzy.utilities import get_cases

# Parameters
N = 10
'''
The amount of parts for cross-validation: (N-1)/Nth of the dataset will 
be used for training and the other part for testing
'''

t_norm = fc.FuzzySet.t_norm_min  # t_norm
implicator = fc.FuzzySet.implicator_kd  # implicator
alpha = 1  # alpha value used in the indiscernibility function
t_norm_str = "minimum"
impl_str = "kd"

# Import data
base_path = Path.cwd().parents[0]
data_path = base_path / 'data'  # Path of data
dataset = 'crx.dat'  # Name of dataset
cases = get_cases(data_path / dataset)  # List of all cases in the dataset

nosum = True
ming = 0.9


# Divide data into N random parts
n = len(cases)
cases_per_part = n // N  # Amount of cases each part of the data will have for cross-validation
lst = [i for i in range(n)]  # List of indices for each case
shuffle(lst)  # Shuffle the list
parts = [lst[i * cases_per_part:(i + 1) * cases_per_part] for i in range(N)]  # Divide lst into N parts
parts[N - 1] += lst[N * cases_per_part:n]  # If N does not divide n, add remaining cases to last part

# Create file to store all output data
path = base_path / 'output'  # Path of dataset
filename = dataset
if nosum:
    filename = "nosum_" + filename
if ming < 1:
    filename = "ming" + str(round(ming, 2)) + "_" + filename

f_out = open(path / filename, 'w')  # Create new file
f_out.write(f"### Dataset: {filename}\n")  # Store name of dataset
f_out.write(
    f"# N = {N}, a = {alpha}, t-norm = {t_norm_str}, implicator = {impl_str}\n")  # Store test-dependent variables
f_out.write(f"{str(lst)}\n")  # Store randomised indices

toggle = True

print('output file created, starting test')

# Start test
accuracies = []
for i in range(N):
    # Separate data into train and test set
    test_part = parts[i]  # The part used for testing
    train_parts = [index for index in lst if index not in test_part]  # The remaining parts for training
    train_data = [cases[index] for index in train_parts]  # The training dataset
    test_data = [cases[index] for index in test_part]  # The testing dataset

    # Create decision system from training data
    a = len(train_data[0])
    ds = fc.DecisionSystem()
    ds.t_norm = t_norm
    ds.implicator = implicator
    ds.cases = [x.copy() for x in train_data]  # Set the cases of the decision system to be the training dataset
    ds.decision = a - 1  # The decision value is the last index of each case
    ds.setup_variables()  # This function automatically sets up all other variables

    # Induce rules from the training data
    concept_keys = list(ds.concepts.keys())  # Get all concept values
    certain_rules = {}  # Dictionary of certain rules for each concept
    possible_rules = {}  # Dictionary of possible rules for each concept
    g_cons_C = {}  # Dictionary of the g-consistency of C
    g_cons_cr = {}  # Dictionary of the g-consistency of C' for the certain rules
    g_cons_pr = {}  # Dictionary of the g-consistency of C' for the possible rules
    for key in concept_keys:  # For each concept, induce a set of rules
        concept = set(ds.concepts[key])
        cr, pr, g_cr, g_pr = fuzzy_rough_lem2.FuzzyRoughLEM2(ds, concept, key, a=alpha)
        certain_rules[key] = cr
        possible_rules[key] = pr
        g_cons_C[key] = ds.g_cons(concept, alpha=alpha)
        g_cons_cr[key] = g_cr
        g_cons_pr[key] = g_pr

    # Store data
    if toggle:
        f_out.write(f"{concept_keys}\n")  # Store concepts
        toggle = False
    f_out.write(f"### Batch: {i}\n")
    f_out.write(f"{ds.attr_stdev}\n")  # Store standard deviations for this data
    f_out.write("### Certain rules\n")
    for key in concept_keys:
        f_out.write(f"# Concept: {key}, g(C) = {g_cons_C[key]}, g(C')) = {g_cons_cr[key]}\n")
        for rule in certain_rules[key]:
            f_out.write(f"{str(rule)}\n")

    f_out.write("### Possible rules\n")
    for key in concept_keys:
        f_out.write(f"# Concept: {key}, , g(C) = {g_cons_C[key]}, g(C') = {g_cons_pr[key]}\n")
        for rule in possible_rules[key]:
            f_out.write(f"{str(rule)}\n")

    # Test the accuracy of the induced rules based on the test data
    cr_accuracy = [0 for i in range(len(test_data))]  # accuracy of the certain rules
    pr_accuracy = [0 for i in range(len(test_data))]  # accuracy of the possible rules

    counter = 0
    for case in test_data:
        # Check, according to the induced rules, which concept this case belongs to most
        max_concept = concept_keys[0]
        max_concept_match = 0
        for key in concept_keys:
            max_rule_match = 0
            # Go over each rule for this concept to see how well it matches with case
            for F in certain_rules[key]:
                # F = {(a_1, v_1), ..., (a_k, v_k)};
                T = 1  # T(R_a1v1(case), ..., R_akvk(case)) with T the t-norm
                for pair in F:
                    T = ds.t_norm(T, ds.R_av_case(pair[0], pair[1], case, alpha=alpha))
                    # Update t-norm
                if T > max_rule_match:
                    max_rule_match = T
            if max_rule_match > max_concept_match:
                max_concept_match = max_rule_match
                max_concept = key
        if max_concept == case[-1]:
            counter += 1
    print(
        f"Correct: {counter}, incorrect: {len(test_data) - counter} for a total accuracy " +
        f"of {round(counter / len(test_data) * 100, 2)}%")
    accuracies.append(round(counter / len(test_data), 4))

print(f"Average accuracy: {round(sum(accuracies) / N, 4)}; all accuracies: {accuracies}")

f_out.close()  # Close output-file
