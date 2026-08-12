import copy

import torch

from trellis.trainers.flow_matching.flow_matching_ControlNet import (
    FlowMatchingTrainer,
    ImageConditionedFlowMatchingCFGTrainer_ControlNet,
)
from trellis.trainers.utils import make_master_params


class _TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([1.0, 2.0]))

    def convert_to_fp16(self):
        self.half()


def _trainer(fp16_mode, is_master=True):
    """绕开完整 Trainer/GPU 初始化，仅覆盖 finetune_from 的参数同步契约。"""
    trainer = object.__new__(ImageConditionedFlowMatchingCFGTrainer_ControlNet)
    model = _TinyModel()
    trainer.models = {"denoiser": model}
    trainer.model_params = list(model.parameters())
    trainer.fp16_mode = fp16_mode
    trainer.master_params = (
        make_master_params(trainer.model_params)
        if fp16_mode == "inflat_all"
        else trainer.model_params
    )
    trainer.is_master = is_master
    trainer.ema_rate = [0.9, 0.99]
    if is_master:
        trainer.ema_params = [
            copy.deepcopy(trainer.master_params) for _ in trainer.ema_rate
        ]
    trainer.world_size = 1
    trainer.device = torch.device("cpu")
    return trainer


def test_controlnet_finetune_initializes_every_ema_from_loaded_master(
    monkeypatch,
):
    loaded = {"weight": torch.tensor([7.0, 9.0])}
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: loaded)

    for fp16_mode in ("inflat_all", "amp", None):
        trainer = _trainer(fp16_mode)
        trainer.finetune_from({"denoiser": "base.pt"})

        for ema_params in trainer.ema_params:
            assert len(ema_params) == len(trainer.master_params)
            for ema_param, master_param in zip(
                ema_params, trainer.master_params
            ):
                torch.testing.assert_close(ema_param, master_param)


def test_controlnet_finetune_does_not_access_ema_on_non_master(monkeypatch):
    loaded = {"weight": torch.tensor([3.0, 4.0])}
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: loaded)
    trainer = _trainer(None, is_master=False)

    trainer.finetune_from({"denoiser": "base.pt"})

    assert not hasattr(trainer, "ema_params")
    torch.testing.assert_close(trainer.models["denoiser"].weight, loaded["weight"])


def test_controlnet_snapshot_temporarily_uses_eval_and_restores_state(
    monkeypatch,
):
    trainer = object.__new__(ImageConditionedFlowMatchingCFGTrainer_ControlNet)
    training_model = _TinyModel().train()
    eval_model = _TinyModel().eval()
    trainer.models = {"training": training_model, "eval": eval_model}

    def fake_snapshot(self, *args, **kwargs):
        assert all(not model.training for model in self.models.values())
        return {"ok": True}

    monkeypatch.setattr(FlowMatchingTrainer, "run_snapshot", fake_snapshot)

    assert trainer.run_snapshot(1) == {"ok": True}
    assert training_model.training is True
    assert eval_model.training is False


def test_controlnet_snapshot_restores_state_after_sampling_error(monkeypatch):
    trainer = object.__new__(ImageConditionedFlowMatchingCFGTrainer_ControlNet)
    model = _TinyModel().train()
    trainer.models = {"denoiser": model}

    def fail_snapshot(self, *args, **kwargs):
        assert model.training is False
        raise RuntimeError("sampling failed")

    monkeypatch.setattr(FlowMatchingTrainer, "run_snapshot", fail_snapshot)

    try:
        trainer.run_snapshot(1)
    except RuntimeError as error:
        assert str(error) == "sampling failed"
    else:
        raise AssertionError("snapshot error was not propagated")
    assert model.training is True


def test_snapshot_inference_prepares_raw_control_once(monkeypatch):
    trainer = object.__new__(ImageConditionedFlowMatchingCFGTrainer_ControlNet)

    class TinyControlModel(_TinyModel):
        @property
        def device(self):
            return self.weight.device

        def __init__(self):
            super().__init__()
            self.prepare_calls = 0

        def prepare_control(self, control, *, batch_size):
            self.prepare_calls += 1
            assert batch_size == 2
            return torch.zeros(1, 8, 4)

        def validate_prepared_control(
            self, prepared_control, *, batch_size, device
        ):
            assert prepared_control.shape == (1, 8, 4)
            assert batch_size == 2
            assert device == self.device

    denoiser = TinyControlModel()
    trainer.models = {"denoiser": denoiser}
    # 避免在这个小型契约测试中初始化真实 DINO，仅保留父类 MRO 的参数透传。
    trainer.encode_image = lambda cond: cond

    args = trainer.get_inference_cond(
        cond=torch.zeros(2, 3, 4),
        control=torch.zeros(2, 1, 4, 4, 4),
        control_scale=0.5,
    )

    assert denoiser.prepare_calls == 1
    assert "control" not in args
    assert args["prepared_control"].shape == (1, 8, 4)
    assert args["control_scale"] == 0.5
