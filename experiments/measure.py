"""
Measurements of the simulated protocol: duration per client and round, bytes sent per round and peak memory per round.

The clients run one after the other in one process, so every client is timed separately. Bytes are counted for an
encoding of the messages, not measured on a network:
- a rule (RulePayload) as compact JSON of RulePayload.to_dict,
- a binary matrix (coverage) as 1 bit per entry,
- other numbers as 8 bytes each,
- labels that are not numbers as JSON.
Memory is the peak of the memory traced by tracemalloc during the step, minus what was traced when the step started.
It includes NumPy arrays. Tracing slows Python down, so durations measured with tracing on are not comparable to
durations without it.
"""
import json
import tracemalloc
from dataclasses import dataclass, field
from math import ceil
from time import perf_counter
from typing import Callable

import numpy as np

from fuzzyroughrules.federated.payload import RulePayload
from fuzzyroughrules.federated.rounds import PerClient


def message_bytes(message) -> int:
    if message is None:
        return 0
    if isinstance(message, RulePayload):
        return len(json.dumps(message.to_dict(), separators=(",", ":")).encode())
    if isinstance(message, np.ndarray):
        if message.ndim == 2 and np.all((message == 0) | (message == 1)):
            return ceil(message.size / 8)
        if message.dtype.kind in "biuf":
            return 8 * message.size
        return len(json.dumps(message.tolist(), separators=(",", ":")).encode())
    if isinstance(message, dict):
        return sum(8 + message_bytes(value) for value in message.values())
    if isinstance(message, (list, tuple, set)):
        return sum(message_bytes(item) for item in message)
    if isinstance(message, (bool, int, float, np.number)):
        return 8
    if isinstance(message, str):
        return len(json.dumps(message).encode())
    raise TypeError(f"No size for a message of type {type(message)}.")


@dataclass
class RoundStats:
    calls: list[list[float]] = field(default_factory=list)  # for each call, the duration of every client
    server_s: float = 0.
    bytes_up: int = 0
    bytes_down: int = 0
    peak_extra_mb: float = 0.


class RoundRecorder:
    """
    A ClientMap and ServerStep (see fuzzyroughrules.federated.rounds) that records every round. A round that is run
    more than once, e.g. a repair round, adds up.
    """

    def __init__(self, trace_memory: bool = False, prefix: str = "", stats: dict[str, RoundStats] = None):
        self.trace_memory = trace_memory
        self.prefix = prefix
        self.stats = {} if stats is None else stats
        if trace_memory and not tracemalloc.is_tracing():
            tracemalloc.start()

    def prefixed(self, prefix: str) -> "RoundRecorder":
        return RoundRecorder(self.trace_memory, self.prefix + prefix, self.stats)

    def close(self):
        if self.trace_memory and tracemalloc.is_tracing():
            tracemalloc.stop()

    def _measured(self, step: Callable[[], object]) -> tuple[object, float, float]:
        if self.trace_memory:
            tracemalloc.reset_peak()
            start_memory = tracemalloc.get_traced_memory()[0]
        start = perf_counter()
        result = step()
        duration = perf_counter() - start
        peak = tracemalloc.get_traced_memory()[1] - start_memory if self.trace_memory else 0
        return result, duration, peak / 2 ** 20

    def __call__(self, round_name: str, clients: list, step: Callable, broadcast=None) -> list:
        stats = self.stats.setdefault(self.prefix + round_name, RoundStats())
        results, durations = [], []
        for index, client in enumerate(clients):
            result, duration, peak = self._measured(lambda: step(index, client))
            results.append(result)
            durations.append(duration)
            stats.peak_extra_mb = max(stats.peak_extra_mb, peak)
            stats.bytes_up += message_bytes(result)
            stats.bytes_down += message_bytes(broadcast[index] if isinstance(broadcast, PerClient) else broadcast)
        stats.calls.append(durations)
        return results

    def server(self, step_name: str, step: Callable[[], object]) -> object:
        stats = self.stats.setdefault(self.prefix + step_name, RoundStats())
        result, duration, peak = self._measured(step)
        stats.server_s += duration
        stats.peak_extra_mb = max(stats.peak_extra_mb, peak)
        return result

    def names(self, prefix: str) -> list[str]:
        return [name for name in self.stats if name.startswith(prefix)]

    def totals(self, names: list[str]) -> dict:
        """
        :param names: the rounds that a model needs
        :return: training time (see training_times in horizontal_fl_frri), number of rounds with communication,
        bytes sent by the clients and by the server, and the peak memory of the rounds
        """
        stats = [self.stats[name] for name in names if name in self.stats]
        calls = [durations for s in stats for durations in s.calls]
        server = sum(s.server_s for s in stats)
        return {
            "train_s_parallel": sum(max(durations) for durations in calls) + server,
            "train_s_sequential": sum(sum(durations) for durations in calls) + server,
            "n_client_rounds": len(calls),
            "bytes_up": sum(s.bytes_up for s in stats),
            "bytes_down": sum(s.bytes_down for s in stats),
            "peak_extra_mb": max((s.peak_extra_mb for s in stats), default=0.),
        }

    def table(self, n_clients: int) -> list[dict]:
        rows = []
        for name, s in self.stats.items():
            row = {"round": name, "n_calls": len(s.calls)}
            if s.calls:
                row.update({f"client_{i}_s": sum(durations[i] for durations in s.calls) for i in range(n_clients)})
            if s.server_s:
                row["server_s"] = s.server_s
            row.update({"bytes_up": s.bytes_up, "bytes_down": s.bytes_down, "peak_extra_mb": s.peak_extra_mb})
            rows.append(row)
        return rows
