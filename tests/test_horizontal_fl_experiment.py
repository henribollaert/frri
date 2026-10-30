from pathlib import Path

import pandas as pd
import pytest
from sklearn.datasets import load_wine

from experiments.horizontal_fl_frri import log_run, run, training_times


def test_training_times_use_the_slowest_client_of_each_round():
    times = training_times([[3., 1.], [2., 4.]], server_time=0.5)
    assert times["train_s_parallel"] == 3. + 4. + 0.5
    assert times["train_s_sequential"] == 10. + 0.5


@pytest.fixture(scope="module")
def result():
    data = load_wine()
    return data, run(data.data, data.target, 3, 0)


def test_every_model_gets_consistent_training_times(result):
    _, (models, local_models, rounds, info, _) = result
    assert (models["train_s_parallel"] > 0).all()
    assert (models["train_s_parallel"] <= models["train_s_sequential"]).all()

    central = models[models["model"] == "centralised FRRI"].iloc[0]
    assert central["train_s_parallel"] == central["train_s_sequential"]

    # the set cover uses all the rounds that the union with global confidence uses, and more
    timed = models.set_index("model")
    assert timed.loc["a union local conf", "train_s_sequential"] == local_models["train_s"].sum()
    assert timed.loc["a union local conf", "train_s_parallel"] == local_models["train_s"].max()
    assert timed.loc["d set cover", "train_s_parallel"] > timed.loc["b union global conf", "train_s_parallel"]

    assert {"local fit", "confidence", "coverage", "coverage counts", "set cover (server)", "centralised fit"} == set(
        rounds["round"]
    )
    assert info["run_s"] >= timed["train_s_sequential"].max()


def test_log_file_can_hold_runs_with_a_different_number_of_clients(result, tmp_path):
    data, (models, _, _, _, _) = result
    log_file = Path(tmp_path) / "runs.csv"
    log_run(log_file, models, Path("wine"), data.data, 3, 0)
    fewer_clients = models.drop(columns="test_local_2")
    log_run(log_file, fewer_clients, Path("wine"), data.data, 2, 1)

    logged = pd.read_csv(log_file)
    assert len(logged) == 2 * len(models)
    assert logged["n_clients"].tolist() == [3] * len(models) + [2] * len(models)
    assert not logged["train_s_parallel"].isna().any()
    assert not any(column.startswith("test_local_") for column in logged.columns)
