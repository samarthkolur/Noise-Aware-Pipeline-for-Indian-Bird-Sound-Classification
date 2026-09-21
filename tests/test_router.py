from pipeline.config import RouterConfig
from pipeline.router import BIRD_BAND, NOISE_BAND, UNCERTAIN_BAND, route_batch, route_segment

CFG = RouterConfig(tau_low=0.30, tau_high=0.70)


def test_high_confidence_bird():
    result = route_segment(mlp_prob=0.9, ood_rejected=False, config=CFG)
    assert result.final_band == BIRD_BAND


def test_high_confidence_noise():
    result = route_segment(mlp_prob=0.1, ood_rejected=False, config=CFG)
    assert result.final_band == NOISE_BAND


def test_uncertain_band():
    result = route_segment(mlp_prob=0.5, ood_rejected=False, config=CFG)
    assert result.final_band == UNCERTAIN_BAND


def test_boundary_tau_high_is_bird():
    result = route_segment(mlp_prob=0.70, ood_rejected=False, config=CFG)
    assert result.final_band == BIRD_BAND


def test_boundary_tau_low_is_noise():
    result = route_segment(mlp_prob=0.30, ood_rejected=False, config=CFG)
    assert result.final_band == NOISE_BAND


def test_ood_rejected_routes_to_noise_even_with_high_prob():
    result = route_segment(mlp_prob=0.95, ood_rejected=True, config=CFG)
    assert result.final_band == NOISE_BAND


def test_route_batch_matches_individual_routing():
    probs = [0.9, 0.5, 0.1, 0.95]
    ood_flags = [False, False, False, True]
    batch_results = route_batch(probs, ood_flags, CFG)
    individual_results = [route_segment(p, o, CFG) for p, o in zip(probs, ood_flags, strict=True)]
    assert [r.final_band for r in batch_results] == [r.final_band for r in individual_results]
