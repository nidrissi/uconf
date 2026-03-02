"""Common regression tests shared by cooperad implementations."""

import pytest

from uconf import CooperadProtocol, SurjectionLinearDual


@pytest.mark.parametrize("r", range(1, 5))
def test_cooperad_protocol_surjection_linear_dual(r: int) -> None:
    """Check that SurjectionLinearDual satisfies CooperadProtocol."""

    assert isinstance(
        SurjectionLinearDual(r), CooperadProtocol
    ), "SurjectionLinearDual should satisfy CooperadProtocol."
