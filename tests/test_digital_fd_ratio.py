import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from digital_fd_ratio import ratios  # noqa: E402


def test_fd_ratio_scales_like_inverse_sqrt_h():
    r = ratios(n=200_000)
    assert r[0.01] > 2.0
    assert 2.5 < r[0.001] / r[0.01] < 4.0
