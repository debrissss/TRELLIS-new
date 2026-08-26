import os
import json
import numpy as np

# 测试必须固定使用 CPU 可用的 SDPA，不能受调用者预设 flash_attn 影响。
os.environ["ATTN_BACKEND"] = "sdpa"

import torch

from trellis.models.sparse_structure_flow import SparseStructureFlowModel
from trellis.models.sparse_structure_flow_ControlNet import (
    SparseStructureFlowModel_ControlNet,
)
from trellis.pipelines.samplers.flow_euler import (
    FlowEulerGuidanceIntervalSampler,
    FlowEulerSampler,
    _control_schedule_gate,
    _scheduled_control_scale,
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


def _assert_raises(expected_exception, callback):
    try:
        callback()
    except expected_exception:
        return
    raise AssertionError(
        f"Expected {expected_exception.__name__} to be raised"
    )


def test_smoothstep_control_schedule_gate_and_scale_contract():
    mild = {
        "name": "smoothstep",
        "full_strength_t": 0.65,
        "min_strength_t": 0.25,
        "min_scale": 0.1,
    }
    assert _control_schedule_gate(1.0, mild) == 1.0
    assert _control_schedule_gate(0.65, mild) == 1.0
    assert _control_schedule_gate(0.25, mild) == 0.1
    assert _control_schedule_gate(0.0, mild) == 0.1
    # The midpoint of smoothstep is exactly halfway between both endpoint scales.
    assert abs(_control_schedule_gate(0.45, mild) - 0.55) < 1e-12
    assert abs(_scheduled_control_scale(2.0, 0.45, mild) - 1.1) < 1e-12
    layer_scales = _scheduled_control_scale([1.0, 0.5], 0.45, mild)
    assert abs(layer_scales[0] - 0.55) < 1e-12
    assert abs(layer_scales[1] - 0.275) < 1e-12

    strong = {**mild, "min_scale": 0.0}
    assert _control_schedule_gate(0.0, strong) == 0.0
    assert _scheduled_control_scale([1.0] * 8, 1.0, strong) == [1.0] * 8

    progress_schedule = {
        "name": "smoothstep",
        "domain": "progress",
        "full_until": 0.6,
        "fade_until": 0.85,
        "min_scale": 0.1,
    }
    assert _control_schedule_gate(0.9, progress_schedule, progress=0.0) == 1.0
    assert _control_schedule_gate(0.9, progress_schedule, progress=0.6) == 1.0
    assert _control_schedule_gate(0.9, progress_schedule, progress=0.85) == 0.1
    assert _control_schedule_gate(0.9, progress_schedule, progress=1.0) == 0.1
    assert abs(
        _control_schedule_gate(0.9, progress_schedule, progress=0.725) - 0.55
    ) < 1e-12


def test_smoothstep_control_schedule_validation_contract():
    valid = {"name": "smoothstep"}
    assert _control_schedule_gate(0.65, valid) == 1.0
    _assert_raises(TypeError, lambda: _control_schedule_gate(0.5, []))
    _assert_raises(ValueError, lambda: _control_schedule_gate(0.5, {}))
    _assert_raises(
        ValueError,
        lambda: _control_schedule_gate(
            0.5, {"name": "linear"}
        ),
    )
    _assert_raises(
        ValueError,
        lambda: _control_schedule_gate(
            0.5,
            {
                "name": "smoothstep",
                "min_strength_t": 0.7,
                "full_strength_t": 0.6,
            },
        ),
    )
    _assert_raises(
        ValueError,
        lambda: _control_schedule_gate(
            0.5, {"name": "smoothstep", "min_scale": -0.1}
        ),
    )
    _assert_raises(
        ValueError,
        lambda: _control_schedule_gate(
            0.5, {"name": "smoothstep", "typo": 0.1}
        ),
    )
    _assert_raises(
        ValueError,
        lambda: _control_schedule_gate(
            0.5, {"name": "smoothstep", "domain": "progress"}
        ),
    )
    _assert_raises(
        ValueError,
        lambda: _control_schedule_gate(
            0.5,
            {
                "name": "smoothstep",
                "domain": "progress",
                "full_until": 0.9,
                "fade_until": 0.8,
            },
            progress=0.5,
        ),
    )


def test_euler_applies_schedule_after_timestep_rescaling_boundary():
    sampler = FlowEulerSampler(sigma_min=1e-5)
    captured = {}

    def model(x_t, t, cond, **kwargs):
        captured["t"] = t
        captured["control_scale"] = kwargs["control_scale"]
        assert "control_schedule" not in kwargs
        return torch.zeros_like(x_t)

    sampler._inference_model(
        model,
        torch.zeros(1, 1, 1, 1, 1),
        0.45,
        cond=torch.zeros(1, 1),
        control_scale=2.0,
        control_schedule={
            "name": "smoothstep",
            "full_strength_t": 0.65,
            "min_strength_t": 0.25,
            "min_scale": 0.1,
        },
    )

    torch.testing.assert_close(captured["t"], torch.tensor([450.0]))
    assert abs(captured["control_scale"] - 1.1) < 1e-12


def test_cfg_multistep_schedule_is_shared_and_never_compounds():
    sampler = FlowEulerGuidanceIntervalSampler(sigma_min=1e-5)
    calls = []

    def model(x_t, t, cond, **kwargs):
        calls.append({
            "t": float(t[0]),
            "branch": "positive" if float(cond[0, 0]) == 1.0 else "negative",
            "control_scale": float(kwargs["control_scale"]),
        })
        return torch.zeros_like(x_t)

    result = sampler.sample(
        model,
        torch.zeros(1, 1, 1, 1, 1),
        cond=torch.ones(1, 1),
        neg_cond=torch.zeros(1, 1),
        steps=4,
        rescale_t=1.0,
        cfg_strength=3.0,
        cfg_interval=(0.0, 1.0),
        verbose=False,
        control_scale=2.0,
        control_schedule={
            "name": "smoothstep",
            "domain": "flow_t",
            "full_strength_t": 0.8,
            "min_strength_t": 0.2,
            "min_scale": 0.1,
        },
    )

    trace = result.control_schedule_trace
    assert len(trace) == 4
    assert len(calls) == 8
    for step_index, trace_row in enumerate(trace):
        positive, negative = calls[2 * step_index:2 * step_index + 2]
        expected_scale = 2.0 * trace_row["gate"]
        assert positive["branch"] == "positive"
        assert negative["branch"] == "negative"
        assert abs(positive["control_scale"] - expected_scale) < 1e-12
        assert abs(negative["control_scale"] - expected_scale) < 1e-12
        assert abs(
            trace_row["effective_control_scale"] - expected_scale
        ) < 1e-12
    # A cumulative implementation would incorrectly multiply the already gated
    # previous-step value. Every expected scale above derives from base=2.0.
    assert calls[-2]["control_scale"] != (
        calls[-4]["control_scale"] * trace[-1]["gate"]
    )


def test_base_checkpoint_has_exact_zero_init_equivalence():
    # 零初始化注入必须保证：即使提供 control，初始输出也与原 flow 逐元素一致。
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
    # 验证冻结边界，并确认首步梯度能进入 zero output Linear。
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


def test_control_condition_reaches_trainable_branch_after_zero_init_warmup():
    """
    双零初始化会延迟条件梯度：第一步先打开 output zero Linear，
    第二步梯度才能进入 control blocks 和 control input projection。
    """
    torch.manual_seed(19)
    base = SparseStructureFlowModel(**_base_args())
    with torch.no_grad():
        torch.nn.init.normal_(base.out_layer.weight, std=0.02)
        torch.nn.init.normal_(base.out_layer.bias, std=0.02)

    model = _control_model()
    model.load_state_dict(base.state_dict())
    model.train()
    optimizer = torch.optim.SGD(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=0.05,
    )

    x, t, cond, control_a = _inputs()
    control_b = 1.0 - control_a
    target = torch.randn_like(x)

    optimizer.zero_grad()
    first_loss = (model(x, t, cond, control=control_a) - target).square().mean()
    first_loss.backward()
    assert torch.count_nonzero(
        model.control_output_layers[0].weight.grad
    ).item() > 0
    input_grad = model.control_input_layer.weight.grad
    assert input_grad is None or torch.count_nonzero(input_grad).item() == 0
    optimizer.step()

    optimizer.zero_grad()
    second_loss = (model(x, t, cond, control=control_a) - target).square().mean()
    second_loss.backward()
    assert torch.count_nonzero(
        model.control_input_layer.weight.grad
    ).item() > 0
    assert any(
        parameter.grad is not None
        and torch.count_nonzero(parameter.grad).item() > 0
        for parameter in model.control_blocks.parameters()
    )
    assert model.control_encoder.input_layer.weight.grad is None
    assert not any(
        parameter.grad is not None
        for parameter in model.blocks.parameters()
    )
    optimizer.step()

    model.eval()
    with torch.no_grad():
        output_a = model(x, t, cond, control=control_a)
        output_b = model(x, t, cond, control=control_b)
        output_single_control = model(x, t, cond, control=control_a[:1])
        output_repeated_control = model(
            x,
            t,
            cond,
            control=control_a[:1].repeat(x.shape[0], 1, 1, 1, 1),
        )
    assert torch.count_nonzero(output_a - output_b).item() > 0
    torch.testing.assert_close(
        output_single_control,
        output_repeated_control,
        rtol=0,
        atol=0,
    )
    _assert_raises(
        ValueError,
        lambda: model(
            x,
            t,
            cond,
            control=control_a[:1].repeat(3, 1, 1, 1, 1),
        ),
    )


def test_dynamic_precision_state_and_control_scale_validation():
    model = _control_model()

    model.convert_to_fp16()
    assert model.use_fp16 is True
    assert model.dtype == torch.float16
    assert model.control_input_layer.weight.dtype == torch.float16
    assert model.control_encoder.dtype == torch.float16

    model.convert_to_fp32()
    assert model.use_fp16 is False
    assert model.dtype == torch.float32
    assert model.control_input_layer.weight.dtype == torch.float32
    assert model.control_encoder.dtype == torch.float32

    assert model._get_control_scales(0.5) == [0.5, 0.5]
    assert model._get_control_scales(torch.tensor(0.25)) == [0.25, 0.25]
    assert model._get_control_scales([0.2, 0.8]) == [0.2, 0.8]

    _assert_raises(ValueError, lambda: model._get_control_scales([1.0, 2.0, 3.0]))
    _assert_raises(ValueError, lambda: model._get_control_scales(float("nan")))
    _assert_raises(ValueError, lambda: model._get_control_scales(float("inf")))
    _assert_raises(ValueError, lambda: model._get_control_scales(torch.ones(2, 1)))
    _assert_raises(TypeError, lambda: model._get_control_scales("1.0"))
    _assert_raises(TypeError, lambda: model._get_control_scales(True))


def test_last_control_residual_is_effective_and_dropout_masks_bias():
    torch.manual_seed(23)
    base = SparseStructureFlowModel(**_base_args())
    with torch.no_grad():
        torch.nn.init.normal_(base.out_layer.weight, std=0.02)
        torch.nn.init.normal_(base.out_layer.bias, std=0.02)

    model = _control_model()
    model.load_state_dict(base.state_dict())
    x, t, cond, control = _inputs()

    with torch.no_grad():
        torch.nn.init.normal_(model.control_input_layer.weight, std=0.02)
        last_output = model.control_output_layers[-1]
        last_output.weight.copy_(torch.eye(model.model_channels))
        last_output.bias.zero_()
        with_last_residual = model.eval()(x, t, cond, control=control)
        last_output.weight.zero_()
        without_last_residual = model(x, t, cond, control=control)
    assert torch.count_nonzero(
        with_last_residual - without_last_residual
    ).item() > 0

    model.control_dropout = 1.0
    with torch.no_grad():
        for output_layer in model.control_output_layers:
            torch.nn.init.normal_(output_layer.weight, std=0.02)
            output_layer.bias.fill_(1.0)
        model.train()
        no_control = model(x, t, cond, control=None)
        dropped_control = model(x, t, cond, control=control)
    torch.testing.assert_close(dropped_control, no_control, rtol=0, atol=0)


def test_pipeline_rejects_plain_flow_and_config_uses_controlnet_prefix():
    from trellis.pipelines.trellis_image_to_3d_ControlNet import (
        TrellisImageTo3DPipeline_ControlNet,
    )

    pipeline = object.__new__(TrellisImageTo3DPipeline_ControlNet)
    pipeline.models = {
        "sparse_structure_flow_model": SparseStructureFlowModel(**_base_args())
    }
    _assert_raises(TypeError, pipeline._validate_controlnet_flow_model)

    controlnet = _control_model()
    pipeline.models["sparse_structure_flow_model"] = controlnet
    assert pipeline._validate_controlnet_flow_model() is controlnet

    with open(
        "configs/pipelines/trellis_image_to_3d_ControlNet.json",
        "r",
    ) as file:
        pipeline_config = json.load(file)
    assert pipeline_config["name"] == "TrellisImageTo3DPipeline_ControlNet"
    assert (
        pipeline_config["args"]["models"]["sparse_structure_flow_model"]
        == "ckpts/ss_flow_ControlNet"
    )
    with open("configs/pipelines/ss_flow_ControlNet.json", "r") as file:
        model_config = json.load(file)
    assert model_config["name"] == "SparseStructureFlowModel_ControlNet"
    assert "control_encoder_ckpt" not in model_config["args"]


def test_complete_controlnet_checkpoint_round_trip():
    # 训练后的完整 checkpoint 必须走严格加载，而不是再次按基础模型扩展。
    model = _control_model()
    restored = _control_model()
    restored.load_state_dict(model.state_dict())

    for expected, actual in zip(model.state_dict().values(), restored.state_dict().values()):
        torch.testing.assert_close(actual, expected)


def test_base_checkpoint_rejects_missing_backbone_key():
    base = SparseStructureFlowModel(**_base_args())
    state = dict(base.state_dict())
    del state["input_layer.weight"]

    _assert_raises(RuntimeError, lambda: _control_model().load_state_dict(state))


def test_base_checkpoint_rejects_unexpected_key():
    base = SparseStructureFlowModel(**_base_args())
    state = dict(base.state_dict())
    state["not_a_backbone.weight"] = torch.zeros(1)

    _assert_raises(RuntimeError, lambda: _control_model().load_state_dict(state))


def test_raw_and_prepared_control_are_mathematically_equivalent():
    model = _control_model().eval()
    base = SparseStructureFlowModel(**_base_args())
    model.load_state_dict(base.state_dict())
    x, t, cond, control = _inputs()
    with torch.no_grad():
        torch.nn.init.normal_(model.control_input_layer.weight, std=0.02)
        torch.nn.init.normal_(model.control_output_layers[0].weight, std=0.02)
        prepared = model.prepare_control(control, batch_size=x.shape[0])
        raw_output = model(x, t, cond, control=control)
        prepared_output = model(
            x, t, cond, prepared_control=prepared
        )

    torch.testing.assert_close(raw_output, prepared_output, rtol=0, atol=0)


def test_prepared_control_batch_one_uses_broadcast_equivalently():
    model = _control_model().eval()
    x, t, cond, control = _inputs()
    prepared_one = model.prepare_control(control[:1], batch_size=x.shape[0])
    prepared_repeated = prepared_one.repeat(x.shape[0], 1, 1)

    with torch.no_grad():
        broadcast_output = model(
            x, t, cond, prepared_control=prepared_one
        )
        repeated_output = model(
            x, t, cond, prepared_control=prepared_repeated
        )
    torch.testing.assert_close(
        broadcast_output, repeated_output, rtol=0, atol=0
    )


def test_raw_and_prepared_control_validation_errors():
    model = _control_model().eval()
    x, t, cond, control = _inputs()
    prepared = model.prepare_control(control, batch_size=x.shape[0])

    _assert_raises(
        ValueError,
        lambda: model(
            x,
            t,
            cond,
            control=control,
            prepared_control=prepared,
        ),
    )
    _assert_raises(
        ValueError,
        lambda: model(x, t, cond, control=control[:, :, :-1]),
    )
    _assert_raises(
        TypeError,
        lambda: model(x, t, cond, control=control.double()),
    )
    _assert_raises(
        ValueError,
        lambda: model(
            x,
            t,
            cond,
            prepared_control=prepared[:, :-1],
        ),
    )
    _assert_raises(
        TypeError,
        lambda: model(
            x,
            t,
            cond,
            prepared_control=prepared.double(),
        ),
    )
    _assert_raises(
        ValueError,
        lambda: model(
            x,
            t,
            cond,
            prepared_control=prepared[:1].repeat(3, 1, 1),
        ),
    )
    _assert_raises(
        ValueError,
        lambda: model._validate_raw_control(
            control, device=torch.device("meta")
        ),
    )
    _assert_raises(
        ValueError,
        lambda: model.validate_prepared_control(
            prepared, device=torch.device("meta")
        ),
    )


def test_pipeline_prepares_raw_control_once_for_all_euler_and_cfg_calls():
    from trellis.pipelines.trellis_image_to_3d_ControlNet import (
        TrellisImageTo3DPipeline_ControlNet,
    )

    class FakeFlow:
        resolution = 2
        in_channels = 1
        device = torch.device("cpu")

        def __init__(self):
            self.prepare_calls = 0
            self.forward_prepared_ids = []
            self.control_encoder = type("Encoder", (), {})()
            self.control_encoder.input_layer = type("InputLayer", (), {})()
            self.control_encoder.input_layer.weight = torch.empty(
                0, dtype=torch.float32
            )

        def _validate_raw_control(self, control, *, batch_size, device):
            assert control.shape == (1, 1, 4, 4, 4)
            assert control.dtype == torch.float32
            assert control.device == device
            assert batch_size == 2

        def prepare_control(self, control, *, batch_size):
            self.prepare_calls += 1
            return torch.zeros(1, 8, 4)

        def validate_prepared_control(
            self, prepared_control, *, batch_size, device
        ):
            assert prepared_control.shape == (1, 8, 4)
            assert prepared_control.dtype == torch.float32
            assert prepared_control.device == device

        def __call__(self, x, prepared_control):
            self.forward_prepared_ids.append(id(prepared_control))
            return x

    class FakeSampler:
        def __init__(self):
            self.control_schedule = None

        def sample(self, model, noise, **kwargs):
            prepared = kwargs["prepared_control"]
            self.control_schedule = kwargs.get("control_schedule")
            # 模拟 3 个 Euler step，每步 CFG 正、负各一次 forward。
            for _ in range(3):
                model(noise, prepared)
                model(noise, prepared)
            return type("Result", (), {"samples": noise})()

    class FakeDecoder:
        def __call__(self, latent):
            return torch.ones(latent.shape[0], 1, 2, 2, 2)

    flow = FakeFlow()
    pipeline = object.__new__(TrellisImageTo3DPipeline_ControlNet)
    pipeline.models = {
        "sparse_structure_flow_model": flow,
        "sparse_structure_decoder": FakeDecoder(),
    }
    sampler = FakeSampler()
    pipeline.sparse_structure_sampler = sampler
    pipeline.sparse_structure_sampler_params = {}
    pipeline._validate_controlnet_flow_model = lambda: flow

    pipeline.sample_sparse_structure(
        {"cond": torch.zeros(1, 1), "neg_cond": torch.zeros(1, 1)},
        num_samples=2,
        # Public API 必须接受 NumPy 4D 单样本，并在进入严格模型 API 前
        # 自动补 batch、转换 dtype 和移动到采样 device。
        control=np.zeros((1, 4, 4, 4), dtype=np.float64),
        control_schedule={
            "name": "smoothstep",
            "full_strength_t": 0.65,
            "min_strength_t": 0.25,
            "min_scale": 0.1,
        },
    )

    assert flow.prepare_calls == 1
    assert len(flow.forward_prepared_ids) == 6
    assert len(set(flow.forward_prepared_ids)) == 1
    assert sampler.control_schedule["name"] == "smoothstep"
    assert sampler.control_schedule["min_scale"] == 0.1
