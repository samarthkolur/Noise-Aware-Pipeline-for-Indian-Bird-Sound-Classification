"""Three-band confidence router (design.md §6.5).

p >= tau_high            -> "bird"       (high-confidence bird)
tau_low < p < tau_high   -> "uncertain"  (human review queue)
p <= tau_low OR OOD      -> "noise"      (confident noise / OOD-rejected)
"""

from __future__ import annotations

from dataclasses import dataclass

from pipeline.config import RouterConfig

BIRD_BAND = "bird"
UNCERTAIN_BAND = "uncertain"
NOISE_BAND = "noise"


@dataclass
class RoutingResult:
    final_band: str
    routed_to: str


def route_segment(mlp_prob: float, ood_rejected: bool, config: RouterConfig) -> RoutingResult:
    """Route one segment to bird/uncertain/noise based on MLP prob + AE OOD flag."""
    if ood_rejected:
        band = NOISE_BAND
    elif mlp_prob >= config.tau_high:
        band = BIRD_BAND
    elif mlp_prob <= config.tau_low:
        band = NOISE_BAND
    else:
        band = UNCERTAIN_BAND
    return RoutingResult(final_band=band, routed_to=f"outputs/{band}/")


def route_batch(
    mlp_probs: list[float], ood_flags: list[bool], config: RouterConfig
) -> list[RoutingResult]:
    return [route_segment(p, ood, config) for p, ood in zip(mlp_probs, ood_flags, strict=True)]
