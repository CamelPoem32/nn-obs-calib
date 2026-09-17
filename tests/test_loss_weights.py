import torch

from obscalib.config import LossWeights
from obscalib.losses import LossComponents, combine_loss_components


def test_loss_components_are_aggregated_with_configurable_weights() -> None:
    rotation = torch.tensor(1.0, requires_grad=True)
    components = LossComponents(
        rotation=rotation,
        translation=torch.tensor(2.0),
        time_offset=torch.tensor(3.0),
        change=torch.tensor(4.0),
        change_time=torch.tensor(5.0),
        prior=torch.tensor(6.0),
        consistency=torch.tensor(7.0),
    )
    weights = LossWeights(
        lambda_rotation=1.0,
        lambda_translation=2.0,
        lambda_time_offset=3.0,
        lambda_change=0.0,
        lambda_change_time=4.0,
        lambda_prior=0.5,
        lambda_consistency=2.0,
    )

    result = combine_loss_components(components, weights)

    assert result.total is not None
    assert torch.allclose(result.total, torch.tensor(51.0))
    result.total.backward()
    assert torch.equal(rotation.grad, torch.tensor(1.0))
