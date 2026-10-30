from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_wine

from experiments.horizontal_fl_frri import log_run, run, training_times
from experiments.measure import RoundRecorder, message_bytes
from fuzzyroughrules.federated.payload import RulePayload


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
    first_round = rounds.set_index("round").loc[["range", "local fit"]].filter(like="client_")
    assert timed.loc["a union local conf", "train_s_sequential"] == pytest.approx(first_round.to_numpy().sum())
    assert timed.loc["a union local conf", "train_s_parallel"] == pytest.approx(first_round.max(axis=1).sum())
    assert timed.loc["a union local conf", "train_s_sequential"] >= local_models["train_s"].sum()
    assert timed.loc["d set cover", "train_s_parallel"] > timed.loc["b union global conf", "train_s_parallel"]
    assert timed.loc["e repair, set cover", "n_client_rounds"] > timed.loc["d set cover", "n_client_rounds"]
    assert (timed["bytes_up"].drop("centralised FRRI") > 0).all()

    assert {"range", "local fit", "confidence", "coverage", "coverage counts", "set cover", "repair: extend",
            "repair: confidence", "centralised fit"} <= set(rounds["round"])
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


def test_older_log_file_with_other_columns_is_rewritten_without_shifting(result, tmp_path):
    data, (models, _, _, _, _) = result
    log_file = Path(tmp_path) / "runs.csv"
    log_run(log_file, models.drop(columns=["bytes_up"]), Path("wine"), data.data, 3, 0)
    log_run(log_file, models, Path("wine"), data.data, 3, 1)
    logged = pd.read_csv(log_file)
    assert logged["bytes_up"].isna().sum() == len(models)
    assert logged["seed"].tolist() == [0] * len(models) + [1] * len(models)
    assert logged["global_test"].tolist() == pytest.approx(models["global_test"].tolist() * 2)


def test_message_bytes_use_the_documented_encoding():
    rule = RulePayload((0, 2), (0.5, 0.25), ("DOMINANT", "INDISCERNIBLE"), (1., 1.), 1, 0.5)
    assert message_bytes(rule) == len(
        '{"attributes":[0,2],"values":[0.5,0.25],"relation_types":["DOMINANT","INDISCERNIBLE"],'
        '"slopes":[1.0,1.0],"label":1,"confidence":0.5}'
    )
    assert message_bytes(np.array([[0., 1., 1.], [1., 0., 0.], [0., 0., 1.]])) == 2
    assert message_bytes(np.array([0.1, 0.2])) == 16
    assert message_bytes([rule, None, (np.zeros(3), np.array(["a", "b"]))]) == message_bytes(rule) + 24 + 9
    assert message_bytes({3: rule, 4: None}) == 16 + message_bytes(rule)


def test_recorder_adds_up_repeated_rounds_and_traces_memory():
    record = RoundRecorder(trace_memory=True)
    try:
        record("step", [1, 2], lambda index, client: np.zeros(100000) + client, broadcast=np.zeros(2))
        record("step", [1, 2], lambda index, client: client, broadcast=np.zeros(2))
        record.server("server", lambda: None)
    finally:
        record.close()
    totals = record.totals(["step", "server", "missing"])
    assert totals["n_client_rounds"] == 2
    assert totals["bytes_up"] == 2 * 800000 + 2 * 8
    assert totals["bytes_down"] == 4 * 16
    assert totals["peak_extra_mb"] > 0.7
    assert totals["train_s_sequential"] >= totals["train_s_parallel"] > 0


def test_non_iid_run_handles_a_client_without_test_local():
    data = load_wine()
    models, local_models, _, _, _ = run(data.data, data.target, 5, 0, dirichlet_alpha=0.5)
    # this split gives a client too small to have test_local data
    assert local_models["own_test_local"].isna().any()
    assert models["mean_test_local"].notna().all()
