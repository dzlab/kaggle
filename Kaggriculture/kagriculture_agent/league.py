"""Pure, deterministic opponent-league sampling helpers.

The sampler deliberately returns the same small match shape used by the
training code, but has no dependency on a trainer or on a policy loader.
Historical checkpoints are grouped into named skill bands.  A checkpoint is
eligible only when its path currently names a regular file; an unavailable
selected band falls back to the current opponent.
"""

from __future__ import annotations

import hashlib
import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


OpponentName = Literal["current", "mixed", "random", "starter", "checkpoint"]
_OPPONENTS = frozenset({"current", "mixed", "random", "starter", "checkpoint"})


def _finite_weight(value: object, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a finite nonnegative number")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be a finite nonnegative number") from exc
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{label} must be a finite nonnegative number")
    return result


@dataclass(frozen=True)
class HistoricalCheckpoint:
    """A historical policy checkpoint and the band assigned to it."""

    path: str | Path
    skill_band: str

    def __post_init__(self) -> None:
        if not isinstance(self.path, (str, Path)) or not str(self.path):
            raise ValueError("checkpoint path must be a non-empty string or Path")
        if not isinstance(self.skill_band, str) or not self.skill_band:
            raise ValueError("checkpoint skill_band must be a non-empty string")
        object.__setattr__(self, "path", str(self.path))

    @property
    def available(self) -> bool:
        """Whether this checkpoint can safely be selected right now."""

        return Path(self.path).is_file()

    @property
    def identity(self) -> str:
        """Return a content identity suitable for rollout provenance."""
        digest = hashlib.sha256()
        with Path(self.path).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return f"checkpoint:{digest.hexdigest()}"


@dataclass(frozen=True)
class SkillBand:
    """Named historical checkpoint band with an optional sampling weight."""

    name: str
    checkpoints: tuple[HistoricalCheckpoint | str | Path, ...] = ()
    weight: float = 1.0

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("skill-band name must be a non-empty string")
        if isinstance(self.checkpoints, (str, bytes)) or not isinstance(self.checkpoints, Sequence):
            raise ValueError("skill-band checkpoints must be a sequence, not a string or bytes")
        weight = _finite_weight(self.weight, "skill-band weight")
        normalized: list[HistoricalCheckpoint] = []
        for checkpoint in tuple(self.checkpoints):
            if isinstance(checkpoint, HistoricalCheckpoint):
                if checkpoint.skill_band != self.name:
                    raise ValueError("checkpoint skill_band must match its containing band")
                normalized.append(checkpoint)
            elif isinstance(checkpoint, (str, Path)) and str(checkpoint):
                normalized.append(HistoricalCheckpoint(checkpoint, self.name))
            else:
                raise ValueError(
                    "skill-band checkpoints must be non-empty strings, Paths, or checkpoint objects"
                )
        object.__setattr__(self, "checkpoints", tuple(normalized))
        object.__setattr__(self, "weight", weight)

    @property
    def available_checkpoints(self) -> tuple[HistoricalCheckpoint, ...]:
        return tuple(checkpoint for checkpoint in self.checkpoints if checkpoint.available)


@dataclass(frozen=True)
class OpponentMatch:
    """A deterministic league draw compatible with the trainer's match shape."""

    opponent: OpponentName | str
    seat: int
    checkpoint: str | None = None
    mixed_opponent: str | None = None
    skill_band: str | None = None
    checkpoint_identity: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.opponent, str) or self.opponent not in _OPPONENTS:
            raise ValueError(f"unsupported opponent: {self.opponent!r}")
        if type(self.seat) is not int or self.seat not in (0, 1):
            raise ValueError("seat must be exactly 0 or 1")
        if self.checkpoint is not None and (not isinstance(self.checkpoint, str) or not self.checkpoint):
            raise ValueError("checkpoint must be a non-empty string when provided")
        if self.mixed_opponent is not None and (
            not isinstance(self.mixed_opponent, str) or not self.mixed_opponent
        ):
            raise ValueError("mixed_opponent must be a non-empty string when provided")
        if self.skill_band is not None and (
            not isinstance(self.skill_band, str) or not self.skill_band
        ):
            raise ValueError("skill_band must be a non-empty string when provided")
        if self.checkpoint_identity is not None and (
            not isinstance(self.checkpoint_identity, str) or not self.checkpoint_identity
        ):
            raise ValueError("checkpoint_identity must be a non-empty string when provided")


DEFAULT_OPPONENT_PROBABILITIES: dict[OpponentName, float] = {
    "current": 0.40,
    "mixed": 0.15,
    "random": 0.10,
    "starter": 0.10,
    "checkpoint": 0.25,
}
DEFAULT_MIXED_OPPONENTS = ("current", "random", "starter")


def _validate_weights(values: Mapping[str, object], label: str) -> dict[str, float]:
    if not isinstance(values, Mapping) or not values:
        raise ValueError(f"{label} must be a non-empty mapping")
    result = {}
    for key, value in values.items():
        if not isinstance(key, str) or not key:
            raise ValueError(f"{label} keys must be non-empty strings")
        result[key] = _finite_weight(value, f"{label}[{key!r}]")
    if sum(result.values()) <= 0.0:
        raise ValueError(f"{label} must contain a positive weight")
    return result


def _rng(seed: int, index: int, stream: str) -> random.Random:
    material = f"{seed}:{index}:{stream}".encode("utf-8")
    digest = hashlib.sha256(material).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def _choose(rng: random.Random, weights: Mapping[str, float]) -> str:
    total = sum(weights.values())
    draw = rng.random() * total
    cumulative = 0.0
    last = next(iter(weights))
    for key, weight in weights.items():
        last = key
        cumulative += weight
        if draw < cumulative:
            return key
    return last


class LeagueSampler:
    """Sample current and historical opponents deterministically.

    ``hard_opponent_weights`` are explicit multipliers applied to band weights;
    leaving them unset preserves the configured band weights.  This keeps
    adaptive weighting an opt-in decision based on an external schedule or
    metric rather than an implicit property of a checkpoint path.
    """

    def __init__(
        self,
        *,
        probabilities: Mapping[str, object] | None = None,
        skill_bands: Mapping[str, SkillBand] = {},
        checkpoint_candidates: Sequence[str | Path] = (),
        band_probabilities: Mapping[str, object] | None = None,
        mixed_opponents: Sequence[str] = DEFAULT_MIXED_OPPONENTS,
        hard_opponent_weights: Mapping[str, object] | None = None,
        hard_opponent_weighting: Mapping[str, object] | None = None,
    ) -> None:
        raw_probabilities = (
            DEFAULT_OPPONENT_PROBABILITIES if probabilities is None else probabilities
        )
        validated = _validate_weights(raw_probabilities, "opponent probabilities")
        unknown = set(validated) - _OPPONENTS
        if unknown:
            raise ValueError(f"unsupported opponent probabilities: {sorted(unknown)}")
        self.probabilities = validated

        if isinstance(checkpoint_candidates, (str, bytes)) or not isinstance(
            checkpoint_candidates, Sequence,
        ):
            raise ValueError("checkpoint_candidates must be a sequence, not a string or bytes")
        if not isinstance(skill_bands, Mapping):
            raise ValueError("skill_bands must be a mapping of names to SkillBand values")
        if checkpoint_candidates:
            if skill_bands:
                raise ValueError("provide skill_bands or checkpoint_candidates, not both")
            skill_bands = {"default": SkillBand("default", tuple(checkpoint_candidates))}
        for name, band in skill_bands.items():
            if not isinstance(name, str) or not name:
                raise ValueError("skill-band mapping keys must be non-empty strings")
            if not isinstance(band, SkillBand):
                raise ValueError("skill-band mapping values must be SkillBand values")
            if band.name != name:
                raise ValueError("skill-band mapping keys must match SkillBand names")
        normalized_bands = tuple(skill_bands.values())
        self.skill_bands = normalized_bands

        if band_probabilities is None:
            selected_band_weights = {band.name: band.weight for band in normalized_bands}
        else:
            selected_band_weights = _validate_weights(band_probabilities, "band probabilities")
            if set(selected_band_weights) != {band.name for band in normalized_bands}:
                raise ValueError("band probabilities must name exactly the configured bands")
        if normalized_bands and sum(selected_band_weights.values()) <= 0.0:
            raise ValueError("band probabilities must contain a positive weight")
        if not normalized_bands and band_probabilities is not None:
            raise ValueError("band probabilities require skill bands")
        self.band_probabilities = selected_band_weights

        if hard_opponent_weights is not None and hard_opponent_weighting is not None:
            raise ValueError("provide only one hard-opponent weighting mapping")
        explicit_hard_weights = hard_opponent_weights or hard_opponent_weighting
        if explicit_hard_weights is None:
            explicit_hard_weights = {}
        else:
            explicit_hard_weights = _validate_weights(
                explicit_hard_weights, "hard-opponent weights",
            )
            if set(explicit_hard_weights) - set(self.band_probabilities):
                raise ValueError("hard-opponent weights must name configured skill bands")
        self.hard_opponent_weights = explicit_hard_weights

        if isinstance(mixed_opponents, (str, bytes)) or not isinstance(mixed_opponents, Sequence):
            raise ValueError("mixed_opponents must be a sequence, not a string or bytes")
        mixed = tuple(mixed_opponents)
        if not mixed or any(not isinstance(name, str) or not name for name in mixed):
            raise ValueError("mixed_opponents must contain non-empty strings")
        self.mixed_opponents = mixed

    def sample(self, index: int, *, seed: int = 0) -> OpponentMatch:
        """Return the match for a stable ``(seed, index)`` coordinate."""
        if type(index) is not int or index < 0:
            raise ValueError("index must be a nonnegative integer")
        if type(seed) is not int:
            raise ValueError("seed must be an integer")
        rng = _rng(seed, index, "opponent")
        selected = _choose(rng, self.probabilities)
        seat = index % 2
        if selected == "checkpoint":
            if not self.skill_bands:
                return OpponentMatch("current", seat)
            band_weights = {
                name: weight * self.hard_opponent_weights.get(name, 1.0)
                for name, weight in self.band_probabilities.items()
            }
            if sum(band_weights.values()) <= 0.0:
                return OpponentMatch("current", seat)
            band_name = _choose(_rng(seed, index, "skill-band"), band_weights)
            band = next(band for band in self.skill_bands if band.name == band_name)
            available = band.available_checkpoints
            if not available:
                return OpponentMatch("current", seat)
            checkpoint = available[_rng(seed, index, "checkpoint").randrange(len(available))]
            try:
                identity = checkpoint.identity
            except OSError:
                return OpponentMatch("current", seat)
            return OpponentMatch(
                "checkpoint", seat, checkpoint=checkpoint.path, skill_band=band.name,
                checkpoint_identity=identity,
            )
        if selected == "mixed":
            choices = self.mixed_opponents
            mixed_opponent = choices[_rng(seed, index, "mixed").randrange(len(choices))]
            return OpponentMatch("mixed", seat, mixed_opponent=mixed_opponent)
        return OpponentMatch(selected, seat)  # type: ignore[arg-type]

    def schedule(self, count: int, *, seed: int = 0) -> list[OpponentMatch]:
        """Return exactly ``count`` weighted matches with alternating seats."""
        if type(count) is not int or count < 1:
            raise ValueError("count must be a positive integer")
        total = sum(self.probabilities.values())
        counts = {
            opponent: int(count * weight / total)
            for opponent, weight in self.probabilities.items()
        }
        missing = count - sum(counts.values())
        remainders = sorted(
            (
                (count * weight / total - counts[opponent], opponent)
                for opponent, weight in self.probabilities.items()
            ),
            key=lambda item: (-item[0], item[1]),
        )
        for _fraction, opponent in remainders[:missing]:
            counts[opponent] += 1

        selected: list[str] = []
        for opponent in self.probabilities:
            selected.extend([opponent] * counts[opponent])
        _rng(seed, count, "schedule").shuffle(selected)

        matches: list[OpponentMatch] = []
        checkpoint_slot = 0
        for index, opponent in enumerate(selected):
            seat = index % 2
            if opponent == "checkpoint":
                if not self.skill_bands:
                    matches.append(OpponentMatch("current", seat))
                    continue
                band_weights = {
                    name: weight * self.hard_opponent_weights.get(name, 1.0)
                    for name, weight in self.band_probabilities.items()
                }
                if sum(band_weights.values()) <= 0.0:
                    matches.append(OpponentMatch("current", seat))
                    continue
                band_name = _choose(_rng(seed, index, "skill-band"), band_weights)
                band = next(band for band in self.skill_bands if band.name == band_name)
                available = band.available_checkpoints
                if not available:
                    matches.append(OpponentMatch("current", seat))
                    continue
                if len(self.skill_bands) == 1 and band.name == "default":
                    checkpoint = available[checkpoint_slot % len(available)]
                    checkpoint_slot += 1
                else:
                    checkpoint = available[
                        _rng(seed, index, "checkpoint").randrange(len(available))
                    ]
                try:
                    identity = checkpoint.identity
                except OSError:
                    matches.append(OpponentMatch("current", seat))
                    continue
                matches.append(OpponentMatch(
                    "checkpoint", seat, checkpoint=checkpoint.path,
                    skill_band=band.name, checkpoint_identity=identity,
                ))
            elif opponent == "mixed":
                mixed_opponent = self.mixed_opponents[
                    _rng(seed, index, "mixed").randrange(len(self.mixed_opponents))
                ]
                matches.append(OpponentMatch(
                    "mixed", seat, mixed_opponent=mixed_opponent,
                ))
            else:
                matches.append(OpponentMatch(opponent, seat))  # type: ignore[arg-type]
        return matches


def _resolve_sampler(
    sampler: LeagueSampler | None,
    sampler_kwargs: Mapping[str, object],
) -> LeagueSampler:
    if sampler is None:
        return LeagueSampler(**sampler_kwargs)
    if not isinstance(sampler, LeagueSampler):
        raise ValueError("sampler must be a LeagueSampler or None")
    if sampler_kwargs:
        raise ValueError("sampler and sampler configuration cannot be combined")
    return sampler


def sample_schedule(
    count: int,
    *,
    seed: int = 0,
    sampler: LeagueSampler | None = None,
    **sampler_kwargs: object,
) -> list[OpponentMatch]:
    """Convenience wrapper around :class:`LeagueSampler.schedule`."""
    selected = _resolve_sampler(sampler, sampler_kwargs)
    return selected.schedule(count, seed=seed)


def sample_match(
    index: int,
    *,
    seed: int = 0,
    sampler: LeagueSampler | None = None,
    **sampler_kwargs: object,
) -> OpponentMatch:
    """Convenience wrapper around :class:`LeagueSampler.sample`."""
    selected = _resolve_sampler(sampler, sampler_kwargs)
    return selected.sample(index, seed=seed)
