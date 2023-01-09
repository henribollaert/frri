

def convert_str(s):
    # Convert a string s to int if it is an integer, float if float, otherwise leave it as string
    try:
        b = float(s)
        if b.is_integer():
            b = int(b)
    except (TypeError, ValueError):
        b = s
    return b


def get_cases(data_path):
    cases = []
    f = open(data_path, 'r')  # Open the file, extract all cases and then close it
    for line in f:
        if line[0] == '@':
            continue
        line = line.replace(' ', '')
        line = line.replace('\t', '')
        line = line.strip('\n')
        case = line.split(',')
        case = [convert_str(val) for val in case[:-1]] + [case[-1]]
        cases.append(case)
    f.close()
    return cases


def get_pairs_from_string(s):
    pairs = set()
    pairs_str = []
    # Get all pairs (as a string) in s
    prev_index = 0
    for index in range(4, len(s)):
        if s[index:index + 2] == "),":
            p = s[prev_index:index + 1]
            prev_index = index + 3
            pairs_str.append(p)
        pairs_str.append(s[prev_index:])

    # Convert strings to pairs
    for p in pairs_str:
        p = p.strip('()')
        p = p.replace(' ', '')
        p = p.split(',')
        a = int(p[0])
        v = convert_str(p[1])
        pairs.add((a, v))
    return pairs
