"""Tests for named calibration-loss aggregation."""

import torch

from obscalib.config import LossWeights
from obscalib.losses import LossComponents, combine_loss_components


def test_loss_components_are_aggregated_with_configurable_weights() -> None:
    rotation = torch.tensor(1.0, requires_grad=True)
    translation = torch.tensor(2.0, requires_grad=True)
    time_offset = torch.tensor(3.0, requires_grad=True)
    change_event = torch.tensor(4.0, requires_grad=True)
    change_time = torch.tensor(5.0, requires_grad=True)
    consistency = torch.tensor(6.0, requires_grad=True)

    components = LossComponents(rotation=rotation, translation=translation, time_offset=time_offset, change_event=change_event, change_time=change_time, consistency=consistency)

    weights = LossWeights(lambda_rotation=1.0, lambda_translation=2.0, lambda_time_offset=3.0, lambda_change_event=0.5, lambda_change_time=4.0, lambda_consistency=0.0)

    result = combine_loss_components(components, weights)

    # 1*1 + 2*2 + 3*3 + 0.5*4 + 4*5 + 0*6 = 36
    assert result.total is not None
    assert torch.allclose(result.total, torch.tensor(36.0))

    result.total.backward()

    torch.testing.assert_close(rotation.grad, torch.tensor(1.0))
    torch.testing.assert_close(translation.grad, torch.tensor(2.0))
    torch.testing.assert_close(time_offset.grad, torch.tensor(3.0))
    torch.testing.assert_close(change_event.grad, torch.tensor(0.5))
    torch.testing.assert_close(change_time.grad, torch.tensor(4.0))
    torch.testing.assert_close(consistency.grad, torch.tensor(0.0))


def test_zero_weight_disables_loss_component_without_removing_it() -> None:
    consistency = torch.tensor(7.0, requires_grad=True)

    components = LossComponents(rotation=torch.tensor(0.0), translation=torch.tensor(0.0), time_offset=torch.tensor(0.0), change_event=torch.tensor(0.0), change_time=torch.tensor(0.0), consistency=consistency)

    weights = LossWeights(lambda_rotation=0.0, lambda_translation=0.0, lambda_time_offset=0.0, lambda_change_event=0.0, lambda_change_time=0.0, lambda_consistency=0.0)

    result = combine_loss_components(components, weights)

    assert result.total is not None
    torch.testing.assert_close(result.total, torch.tensor(0.0))

    result.total.backward()

    assert consistency.grad is not None
    torch.testing.assert_close(consistency.grad, torch.tensor(0.0))