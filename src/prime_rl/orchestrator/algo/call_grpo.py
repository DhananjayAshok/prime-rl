from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Hashable, Literal

import verifiers.v1 as vf

from prime_rl.configs.algorithm import CallGRPOAlgoConfig
from prime_rl.orchestrator.algo.base import Algorithm, iter_trainable_traces
from prime_rl.orchestrator.algo.routing import assign_advantages, trainable_nodes

if TYPE_CHECKING:
    from prime_rl.orchestrator.clients import InferenceClient

Baseline = Literal["unit", "unit_tag", "within_tag", "within_trace", "hybrid"]
UNIT_BASELINES = ("unit", "unit_tag")


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _peers(agents: list[str]) -> dict[str, list[int]]:
    """The indices of each agent's rollouts in the group."""
    peers: dict[str, list[int]] = defaultdict(list)
    for i, agent in enumerate(agents):
        peers[agent].append(i)
    return peers


def _within_advantages(*, rewards: list[list[float]], keys: list[list[Hashable]]) -> list[list[float]]:
    """A call's reward minus the mean reward of the calls with the same key in its own rollout."""
    out = []
    for rs, ks in zip(rewards, keys, strict=True):
        by_key: dict[Hashable, list[float]] = defaultdict(list)
        for r, k in zip(rs, ks, strict=True):
            by_key[k].append(r)
        out.append([r - _mean(by_key[k]) for r, k in zip(rs, ks, strict=True)])
    return out


def _unit_advantages(
    *, rewards: list[list[float]], keys: list[list[Hashable]], agents: list[str]
) -> list[list[float]]:
    """A call's reward minus the mean, over the same agent's rollouts in the group (its own
    included), of their reward for the call's key. A rollout's reward for a key is the mean of
    its calls with that key: they share a reward unless the env penalized one call alone (an
    invalid answer). A call whose key no other rollout has gets advantage 0."""
    key_rewards: list[dict[Hashable, float]] = []
    for rs, ks in zip(rewards, keys, strict=True):
        by_key: dict[Hashable, list[float]] = defaultdict(list)
        for r, k in zip(rs, ks, strict=True):
            by_key[k].append(r)
        key_rewards.append({k: _mean(v) for k, v in by_key.items()})
    peers = _peers(agents)

    out = []
    for i, (rs, ks) in enumerate(zip(rewards, keys, strict=True)):
        others = [j for j in peers[agents[i]] if j != i]
        advantages = []
        for r, k in zip(rs, ks, strict=True):
            same_key = [key_rewards[j][k] for j in others if k in key_rewards[j]]
            advantages.append(r - _mean([key_rewards[i][k], *same_key]) if same_key else 0.0)
        out.append(advantages)
    return out


def call_advantages(
    *,
    rewards: list[list[float]],
    units: list[list[Hashable]] | None,
    tags: list[list[Hashable]],
    agents: list[str],
    baseline: Baseline,
    game_rewards: list[float] | None = None,
    hybrid_within: Literal["within_tag", "within_trace"] = "within_tag",
    game_weight: float = 1.0,
) -> list[list[float]]:
    """Per-call advantages for one group: rollout i has calls with ``rewards[i]``,
    ``units[i]`` (``unit`` and ``unit_tag`` only) and ``tags[i]``, scalar reward
    ``game_rewards[i]`` (``hybrid`` only) and belongs to agent ``agents[i]``. See
    ``CallGRPOAlgoConfig.baseline``."""
    if baseline in UNIT_BASELINES and units is None:
        raise ValueError(f"call_grpo baseline {baseline!r} needs units")
    if baseline == "unit":
        return _unit_advantages(rewards=rewards, keys=units, agents=agents)
    if baseline == "unit_tag":
        keys = [list(zip(us, ts, strict=True)) for us, ts in zip(units, tags, strict=True)]
        return _unit_advantages(rewards=rewards, keys=keys, agents=agents)
    if baseline == "within_tag":
        return _within_advantages(rewards=rewards, keys=tags)
    if baseline == "within_trace":
        return _within_advantages(rewards=rewards, keys=[[None] * len(rs) for rs in rewards])
    if baseline == "hybrid":
        if game_rewards is None:
            raise ValueError("call_grpo baseline 'hybrid' needs game_rewards")
        if hybrid_within not in ("within_tag", "within_trace"):
            raise ValueError(f"unknown call_grpo hybrid_within {hybrid_within!r}")
        within = call_advantages(rewards=rewards, units=units, tags=tags, agents=agents, baseline=hybrid_within)
        peers = _peers(agents)
        out = []
        for i, call_advs in enumerate(within):
            game_adv = game_rewards[i] - _mean([game_rewards[j] for j in peers[agents[i]]])
            out.append([game_weight * game_adv + a for a in call_advs])
        return out
    raise ValueError(f"unknown call_grpo baseline {baseline!r}")


def keep_credited(
    *, advantages: list[list[float]], tags: list[list[Hashable]], invalid: list[list[bool]], kinds: list[str]
) -> list[list[float]]:
    """``advantages`` with every call zeroed except those whose tag's kind (the part of the tag
    before the first ``:``) is in ``kinds`` and those flagged ``invalid``."""
    return [
        [a if bad or str(t).split(":")[0] in kinds else 0.0 for a, t, bad in zip(advs, ts, inv, strict=True)]
        for advs, ts, inv in zip(advantages, tags, invalid, strict=True)
    ]


class CallGRPOAlgorithm(Algorithm):
    """GRPO with per-call credit: each model call (trainable node) of a trace gets its own
    advantage, its env-provided reward minus a baseline from comparable calls (see
    ``CallGRPOAlgoConfig``). The scalar trace reward is used only by the ``hybrid`` baseline."""

    def __init__(self, config: CallGRPOAlgoConfig, clients: InferenceClient):
        super().__init__(config, clients)
        self.config = config

    def _per_call(self, trace: vf.Trace, key: str, n: int, required: bool) -> list | None:
        """``trace.info[key]``, which must hold one entry per trainable node. None if absent
        and not required."""
        values = trace.info.get(key)
        if values is None:
            if required:
                raise ValueError(
                    f"call_grpo (baseline {self.config.baseline!r}) needs trace.info[{key!r}], one entry "
                    f"per trainable node; trace {trace.id!r} has none"
                )
            return None
        if len(values) != n:
            raise ValueError(
                f"call_grpo: trace.info[{key!r}] of trace {trace.id!r} has {len(values)} entries "
                f"for {n} trainable nodes"
            )
        return list(values)

    async def score_group(self, episodes: list[vf.Episode]) -> None:
        traces = [trace for _, trace in iter_trainable_traces(episodes)]
        needs_units = self.config.baseline in UNIT_BASELINES
        credit_kinds = None if self.config.credit_kinds == "all" else self.config.credit_kinds
        rewards, units, tags, invalid = [], [], [], []
        for trace in traces:
            n = len(trainable_nodes(trace))
            rewards.append([float(r) for r in self._per_call(trace, self.config.rewards_key, n, required=True)])
            units.append(self._per_call(trace, self.config.units_key, n, required=needs_units))
            tags.append(self._per_call(trace, self.config.tags_key, n, required=credit_kinds is not None) or [None] * n)
            if credit_kinds is not None:
                invalid.append([bool(b) for b in self._per_call(trace, self.config.invalid_key, n, required=True)])

        advantages = call_advantages(
            rewards=rewards,
            units=units if needs_units else None,
            tags=tags,
            agents=[trace.agent.name for trace in traces],
            baseline=self.config.baseline,
            game_rewards=[float(trace.reward) for trace in traces],
            hybrid_within=self.config.hybrid_within,
            game_weight=self.config.game_weight,
        )
        if credit_kinds is not None:
            advantages = keep_credited(advantages=advantages, tags=tags, invalid=invalid, kinds=credit_kinds)
        for trace, call_advs in zip(traces, advantages, strict=True):
            per_token: list[float] = []
            for advantage, node in zip(call_advs, trainable_nodes(trace), strict=True):
                per_token += [advantage] * sum(node.mask)
            assign_advantages(trace, per_token)
