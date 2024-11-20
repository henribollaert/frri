from pathlib import Path



def filter_integers_from_file(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            # Split each line into words
            words = line.split()
            if len(words) == 1:
                outfile.write(words[0] + '\n')
            else:
                # Filter words that are integers
                integers = [word for word in words if word.isdigit()]
                # Write integers to the output file, joined by a space
                if len(integers) > 0:
                    outfile.write(', '.join(integers) + '\n')

# Usage example
def main():
    threshold = 70
    input_file = f'/Users/henri/Library/Mobile Documents/iCloud~md~obsidian/Documents/main-vault/uncovered&candidates output MSE threshold {threshold}.md'
    output_file = f'/Users/henri/Library/CloudStorage/OneDrive-Personal/Documents/_Work/PhD Thesis/2022-fuzzylem/relabelling-counts/MSE-CVX-uncertainty-counts-threshold{threshold}.txt'

    # threshold = '10'
    # input_file = f'/Users/henri/Library/Mobile Documents/iCloud~md~obsidian/Documents/main-vault/relabelling output MSE threshold{threshold}.md'
    # output_file = f'/Users/henri/Library/CloudStorage/OneDrive-Personal/Documents/_Work/PhD Thesis/2022-fuzzylem/relabelling-counts/MSE-CVX-relabelling-counts-threshold{threshold}.txt'

    filter_integers_from_file(input_file, output_file)

if __name__ == "__main__":
    main()
