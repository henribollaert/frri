from pathlib import Path
import pandas as pd
import numpy as np

def load_weka_results():
    weka_folder = Path.cwd() / 'weka-results'
    weka_models = {
        'modlem': '&',
        'furia': 'and',
        'ripper': 'and'
    }
    acc_type = 'balanced-accuracy'  # 'weka-accuracy.csv'

    scores = {}
    rules = {}
    attributes = {}

    for name, connection in weka_models.items():
        model_folder = weka_folder / f"{name}-files"
        file = model_folder / f"{name}-{acc_type}.csv"
        scores[name] = pd.read_csv(weka_folder / file, header=None, index_col=0).to_dict()[1]
        rules[name] = {}
        attributes[name] = {}
        for file in model_folder.iterdir():
            if file.name[-3:] == 'txt':
                short_name = file.name[:-4]
                with open(file, 'r') as f:
                    line = f.readline()
                    nrs = []
                    while len(line) > 4:
                        nrs.append(line.count(connection) + 1)
                        line = f.readline()
                rules[name][short_name] = len(nrs)
                attributes[name][short_name] = np.average(nrs)

    return scores, rules, attributes


def get_dataset(
        folder_path: Path,
        keyword: str,
        remove_cat: bool = False,
        get_datatypes: bool = False) -> tuple[np.ndarray, ...]:
    """
    Returns the unique dataset in the folder indicated by folder_path that contains the keyword.
    Categorical features can be removed, and the datatypes can also be returned
    :param folder_path: path where the data is located
    :param keyword: keyword contained in the name of the dataset
    :param remove_cat: should categorical features be removed (except for the decision attribute)
    :param get_datatypes: should we return the
    :return: values of conditional features, values of decision attribute,
    possibly the datatypes of the conditional features
    """
    set_list = [_ for _ in folder_path.iterdir() if keyword in _.name]
    assert len(set_list) == 1, f'{len(set_list)} files with {keyword} in their name.'

    dataset = pd.read_csv(set_list[0], header=None, comment='@')
    if remove_cat:
        nums = [t != 'object' for t in dataset.dtypes]
        nums[-1] = False
        x_dataset = dataset.loc[:, nums]
    else:
        x_dataset = dataset.iloc[:, :-1]
    if get_datatypes:
        result = x_dataset.values, dataset.iloc[:, -1].values, x_dataset.dtypes.values
    else:
        result = x_dataset.values, dataset.iloc[:, -1].values
    return result
