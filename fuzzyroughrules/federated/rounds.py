"""
Extra rounds of the protocol, simulated in one process: rule repair and coverage rounds.

The clients are reached through a ClientMap, which runs one step on every client and returns the results in the order
of the clients. It also gets what the server broadcasts to every client in that round, so that an experiment can count
the bytes. The default runs the clients one after the other; an experiment can pass its own to time or measure a
round. Work on the server goes through a ServerStep in the same way.
"""
from typing import Callable

import numpy as np

from fuzzyroughrules.federated.aggregate import global_confidence, plain_set_cover, set_cover, union_global
from fuzzyroughrules.federated.client import FederatedClient
from fuzzyroughrules.federated.payload import RulePayload

class PerClient(list):
    """
    A broadcast that differs per client: item i is what client i gets.
    """


ClientMap = Callable[..., list]
ServerStep = Callable[[str, Callable[[], object]], object]
REPAIR_TOLERANCE = 1e-6  # the inclusion threshold of FRRI allows the local confidence to be this much too high


def sequential(
        round_name: str,
        clients: list[FederatedClient],
        step: Callable[[int, FederatedClient], object],
        broadcast=None
) -> list:
    """
    :param round_name: name of the round, for measurements
    :param clients: the clients
    :param step: what a client does, given its index and itself
    :param broadcast: what the server sends to every client in this round. Not used here.
    :return: the result of every client
    """
    return [step(index, client) for index, client in enumerate(clients)]


def on_server(step_name: str, step: Callable[[], object]) -> object:
    return step()


def repair_rules(
        clients: list[FederatedClient],
        client_rules: list[list[RulePayload]],
        confidence: np.ndarray,
        max_rounds: int = None,
        tolerance: float = REPAIR_TOLERANCE,
        map_clients: ClientMap = sequential
) -> tuple[list[list[RulePayload]], np.ndarray, list[dict]]:
    """
    Rule repair. While the global confidence of a rule is below its local confidence, the client that owns the rule
    adds one attribute (FederatedClient.extend_rule), and every client returns its minimum for the longer rule. A rule
    stops when its global confidence is back at the local confidence, or when it uses all attributes. Every round
    reveals one more attribute value of the object of each rule that is still being repaired.
    :param clients: the clients, after fit_local
    :param client_rules: the rules of each client, with their local confidence
    :param confidence: the global confidence of the rules, in the order of union_local(client_rules)
    :param max_rounds: maximum number of repair rounds, None for no limit
    :param tolerance: a global confidence this much below the local confidence counts as recovered
    :param map_clients: runs a step on every client
    :return: the repaired rules of each client, still with their local confidence, their global confidence, and one
    log entry per round
    """
    rules = [list(own) for own in client_rules]
    confidence = np.array(confidence, dtype=float)
    positions = [(owner, index) for owner, own in enumerate(rules) for index in range(len(own))]
    active = [p for p, (owner, index) in enumerate(positions)
              if confidence[p] < rules[owner][index].confidence - tolerance]
    log = []
    while active and (max_rounds is None or len(log) < max_rounds):
        # the server tells each owner which of its rules to extend
        extended = map_clients("repair: extend", clients, lambda owner, client: {
            p: client.extend_rule(positions[p][1], rules[owner][positions[p][1]])
            for p in active if positions[p][0] == owner
        }, PerClient([[positions[p][1] for p in active if positions[p][0] == owner] for owner in range(len(rules))]))
        extended = {p: rule for proposals in extended for p, rule in proposals.items() if rule is not None}
        if not extended:
            break
        keys = list(extended)
        payloads = [extended[p] for p in keys]
        minima = map_clients("repair: confidence", clients, lambda _, client: client.local_minima(payloads), payloads)
        new_confidence = global_confidence(minima)
        for p, payload, value in zip(keys, payloads, new_confidence):
            owner, index = positions[p]
            rules[owner][index] = payload
            confidence[p] = value
        active = [p for p in keys if confidence[p] < extended[p].confidence - tolerance]
        log.append({
            "round": len(log) + 1,
            "n_rules_extended": len(keys),
            "n_recovered": len(keys) - len(active),
            "n_still_below": len(active),
        })
    return rules, confidence, log


def confidence_weighted_set_cover(rules: list[RulePayload]) -> np.ndarray:
    """
    Objective that prefers confident rules: the cost of a rule is 2 - its confidence, so between 1 and 2. A rule with
    confidence 1 costs as much as half a rule with confidence 0 more than it.
    """
    return 2 - np.array([rule.confidence for rule in rules])


def coverage_rounds(
        clients: list[FederatedClient],
        client_rules: list[list[RulePayload]],
        confidence: np.ndarray,
        seed: int,
        max_rounds: int = 3,
        objective: Callable[[list[RulePayload]], np.ndarray] = plain_set_cover,
        solver: str = "auto",
        time_limit: float = None,
        map_clients: ClientMap = sequential,
        run_on_server: ServerStep = on_server
) -> tuple[list[RulePayload], int, list[RulePayload], list[dict]]:
    """
    Set cover with coverage rounds. After each set cover, every client sends the FRRI rules of its objects that the
    selected rules do not cover (FederatedClient.uncovered_rules). These new candidates get their global confidence,
    the coverage is computed again and the set cover is solved again.
    :param clients: the clients, after fit_local
    :param client_rules: the candidate rules of each client, with their local confidence
    :param confidence: the global confidence of the candidates, in the order of union_local(client_rules)
    :param seed: seed for shuffling the coverage columns
    :param max_rounds: maximum number of coverage rounds after the first set cover
    :param objective: gives the cost of selecting each rule
    :param solver: solver of the set cover
    :param time_limit: time limit of each set cover
    :param map_clients: runs a step on every client
    :param run_on_server: runs the set cover on the server
    :return: the selected rules, the number of objects that no candidate covers in the last set cover, the candidates
    of the last set cover with their global confidence, and one log entry per set cover
    """
    candidates = [list(own) for own in client_rules]
    for client, own in zip(clients, candidates):
        client.start_coverage_rounds(own)
    confidence = np.array(confidence, dtype=float)
    log = []
    while True:
        rated = union_global(candidates, confidence)
        coverage = np.hstack(map_clients(
            "coverage", clients, lambda index, client: client.coverage_columns(rated, seed + index + 1000 * len(log)),
            rated
        ))
        selected, n_uncovered = run_on_server(
            "set cover", lambda: set_cover(rated, coverage, objective, solver, time_limit)
        )
        entry = {
            "round": len(log),
            "n_candidates": sum(len(own) for own in candidates),
            "n_selected": len(selected),
            # the selection covers every object that a candidate covers
            "n_uncovered": n_uncovered,
            "n_new_candidates": 0,
        }
        log.append(entry)
        if len(log) > max_rounds:
            break
        new = map_clients(
            "coverage: new candidates", clients, lambda _, client: client.uncovered_rules(selected), selected
        )
        entry["n_new_candidates"] = sum(len(own) for own in new)
        if entry["n_new_candidates"] == 0:
            break
        new_union = [rule for own in new for rule in own]
        minima = map_clients(
            "coverage: confidence", clients, lambda _, client: client.local_minima(new_union), new_union
        )
        new_confidence = global_confidence(minima)
        # keep the order of union_local: the rules of client 0 first, then those of client 1, ...
        merged, start, offset = [], 0, 0
        for owner, own in enumerate(candidates):
            merged.append(confidence[start:start + len(own)])
            merged.append(new_confidence[offset:offset + len(new[owner])])
            start += len(own)
            offset += len(new[owner])
            own.extend(new[owner])
        confidence = np.concatenate(merged)
    return selected, n_uncovered, rated, log
