import os

os.environ.setdefault("ATTN_BACKEND", "sdpa")

import torch

from trellis.models.sparse_structure_flow import SparseStructureFlowModel
from trellis.models.sparse_structure_flow_ControlNet import (
    SparseStructureFlowModel_ControlNet,
)


def _base_args():
    return {
        "resolution": 4,
        "in_channels": 4,
        "out_channels": 4,
        "model_channels": 8,
        "cond_channels": 8,
        "num_blocks": 3,
        "num_heads": 2,
        "mlp_ratio": 2,
        "patch_size": 1,
        "pe_mode": "ape",
        "use_fp16": False,
    }


def _control_model():
    return SparseStructureFlowModel_ControlNet(
        **_base_args(),
        control_channels=1,
        control_resolution=8,
        control_encoder_args={
            "in_channels": 1,
            "latent_channels": 4,
            "num_res_blocks": 1,
            "num_res_blocks_middle": 1,
            "channels": [4, 8],
            "use_fp16": False,
        },
        num_control_blocks=2,
        control_dropout=0.0,
        freeze_backbone=True,
    )


def _inputs():
    torch.manual_seed(7)
    x = torch.randn(2, 4, 4, 4, 4)
    t = torch.tensor([100.0, 900.0])
    cond = torch.randn(2, 5, 8)
    control = torch.randint(0, 2, (2, 1, 8, 8, 8)).float()
    return x, t, cond, control


def test_base_checkpoint_has_exact_zero_init_equivalence():
    torch.manual_seed(11)
    base = SparseStructureFlowModel(**_base_args()).eval()
    with torch.no_grad():
        torch.nn.init.normal_(base.out_layer.weight, std=0.02)
        torch.nn.init.normal_(base.out_layer.bias, std=0.02)

    controlnet = _control_model().eval()
    controlnet.load_state_dict(base.state_dict())

    x, t, cond, control = _inputs()
    with torch.no_grad():
        expected = base(x, t, cond)
        actual = controlnet(x, t, cond, control=control)
        disabled = controlnet(x, t, cond, control=control, control_scale=0.0)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(disabled, expected, rtol=0, atol=0)


def test_only_control_branch_is_trainable_and_zero_layers_receive_gradient():
    torch.manual_seed(13)
    base = SparseStructureFlowModel(**_base_args())
    with torch.no_grad():
        torch.nn.init.normal_(base.out_layer.weight, std=0.02)
        torch.nn.init.normal_(base.out_layer.bias, std=0.02)

    model = _control_model()
    model.load_state_dict(base.state_dict())
    model.train()

    trainable = {
        name for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    assert trainable
    assert all(name.startswith("control_") for name in trainable)
    assert not any(name.startswith("control_encoder.") for name in trainable)

    x, t, cond, control = _inputs()
    loss = model(x, t, cond, control=control).square().mean()
    loss.backward()

    output_grad = model.control_output_layers[0].weight.grad
    assert output_grad is not None
    assert torch.count_nonzero(output_grad).item() > 0
    assert model.control_encoder.input_layer.weight.grad is None


def test_complete_controlnet_checkpoint_round_trip():
    model = _control_model()
    restored = _control_model()
    restored.load_state_dict(model.state_dict())

    for expected, actual in zip(model.state_dict().values(), restored.state_dict().values()):
        torch.testing.assert_close(actual, expected)
