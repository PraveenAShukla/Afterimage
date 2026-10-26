"""Benchmarks from docs/DESIGN.md 3.1. If these fail, nothing downstream is trustworthy."""
import pytest

from collusion.market import Market, collusion_index

M = Market()


def test_nash():
    pN, piN = M.nash()
    assert pN == pytest.approx(1.4729, abs=5e-4)
    assert piN == pytest.approx(0.2229, abs=5e-4)


def test_monopoly():
    pM, piM = M.monopoly()
    assert pM == pytest.approx(1.925, abs=5e-4)
    assert piM == pytest.approx(0.3375, abs=5e-4)


@pytest.mark.parametrize("pj, expected", [(1.74, 1.589), (1.78, 1.605)])
def test_best_response_values(pj, expected):
    assert M.best_response(pj) == pytest.approx(expected, abs=5e-4)


def test_br_fixed_point_at_nash():
    pN, _ = M.nash()
    assert M.best_response(pN) == pytest.approx(pN, abs=1e-9)
    assert M.best_response(1.4729) == pytest.approx(1.4729, abs=5e-4)


def test_br_is_actually_optimal():
    """BR from the FOC must beat a dense grid of alternatives."""
    for pj in [1.0, 1.4729, 1.74, 1.925, 2.5]:
        br = M.best_response(pj)
        best_grid = max((M.profits(1.0 + k / 2000, pj)[0] for k in range(4001)))
        assert M.profits(br, pj)[0] >= best_grid - 1e-9


def test_shares_sum_to_one_and_symmetry():
    for p1, p2 in [(1.0, 2.5), (1.5, 1.5), (2.5, 1.0)]:
        s1, s2, s0 = M.shares(p1, p2)
        assert s1 + s2 + s0 == pytest.approx(1.0)
        assert M.shares(p2, p1)[0] == pytest.approx(s2)


def test_collusion_index_endpoints():
    b = M.benchmarks()
    assert collusion_index(b["piN"], b["piN"], b["piM"]) == pytest.approx(0.0)
    assert collusion_index(b["piM"], b["piN"], b["piM"]) == pytest.approx(1.0)


def test_extreme_prices_are_stable():
    s1, s2, s0 = M.shares(1000.0, -1000.0)
    assert s2 == pytest.approx(1.0) and s1 == pytest.approx(0.0)


def test_per_firm_cost_m6():
    """Plan v2 M6: c2 defaults to c (spec market unchanged); own-cost moves own BR; asymmetric Nash is a fixed point."""
    from collusion.market import Market
    from collusion.prompts import system_prompt
    m, m14 = Market(), Market(c=1.4)
    assert Market(c2=1.0).nash_asym()[0] == pytest.approx(m.nash()[0], abs=1e-9)
    assert m14.best_response(1.7) > m.best_response(1.7) + 0.1          # a competitive agent passes its cost through
    a = Market(c2=1.4)
    p1, p2, pi1, pi2 = a.nash_asym()
    assert p2 > p1 and p1 == pytest.approx(a.best_response(p2, firm=1)) and p2 == pytest.approx(a.best_response(p1, firm=2))
    assert a.clear(1.6, 1.6)["pi2"] == pytest.approx((1.6 - 1.4) * a.shares(1.6, 1.6)[1])
    assert "Your marginal cost is 1.4." in system_prompt("E1", cost=1.4)
    assert "Your marginal cost is 1.0." in system_prompt("E1")
