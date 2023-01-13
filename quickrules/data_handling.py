import pandas as pd


def get_dataset(folder_path, keyword, remove_cat=True):
    """
    Returns a dataset from a specified folder with a given keyword in the name, possibly after removing categorical
    features.

    Parameters
    ----------
    folder_path path to folder of the dataset
    keyword     keyword [d*]tra or [d*]tst
    remove_cat  remove categorical features, yes or no?

    Returns
    -------
    tuple containing numpy array containing x values, numpy array containing y values
    """
    set_list = [_ for _ in folder_path.iterdir() if keyword in _.name]
    assert len(set_list) == 1, f'{ len(set_list)} files with {keyword} in their name.'

    dataset = pd.read_csv(set_list[0], header=None, comment='@')
    if remove_cat:
        nums = [t != 'object' for t in dataset.dtypes]
        nums[-1] = False
        x_dataset = dataset.loc[:, nums]
    else:
        x_dataset = dataset.iloc[:, :-1]
    y_dataset = dataset.iloc[:, -1]
    return x_dataset.values, y_dataset.values
