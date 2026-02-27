"""Common regression tests shared by operad implementations."""

import pytest

from uconf import BarrattEccles, OperadProtocol, Surjection


@pytest.mark.parametrize("r", range(1, 5))
def test_operad_protocol(r: int) -> None:
    """Check that Surjection and Barratt-Eccles satisfy the OperadProtocol."""
    assert isinstance(
        Surjection(r), OperadProtocol
    ), "Surjection should satisfy OperadProtocol."
    assert isinstance(
        BarrattEccles(r), OperadProtocol
    ), "Barratt-Eccles should satisfy OperadProtocol."