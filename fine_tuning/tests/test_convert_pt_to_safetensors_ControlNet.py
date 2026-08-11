from pathlib import Path

from fine_tuning import convert_pt_to_safetensors_ControlNet as module


def _config():
    return {
        "name": "SparseStructureFlowModel_ControlNet",
        "args": {
            "resolution": 16,
            "control_encoder_ckpt": "/training-host/encoder.pt",
        },
    }


def test_portable_config_removes_local_control_encoder_checkpoint():
    source = _config()
    portable = module.make_portable_model_config(source)

    assert "control_encoder_ckpt" not in portable["args"]
    assert source["args"]["control_encoder_ckpt"] == "/training-host/encoder.pt"


def test_explicit_deploy_config_is_sanitized(monkeypatch):
    explicit = _config()
    explicit["args"]["control_dropout"] = 0.0
    monkeypatch.setattr(module, "load_json", lambda path: explicit)

    portable = module.get_deploy_model_config(
        _config(), Path("deploy.json")
    )

    assert portable["args"]["control_dropout"] == 0.0
    assert "control_encoder_ckpt" not in portable["args"]


def test_complete_state_validation_rejects_missing_encoder_weight():
    class FakeModel:
        def state_dict(self):
            return {
                "control_encoder.input_layer.weight": object(),
                "control_input_layer.weight": object(),
            }

    try:
        module.validate_complete_controlnet_state(
            FakeModel(), {"control_input_layer.weight": object()}
        )
    except RuntimeError as error:
        assert "control_encoder.input_layer.weight" in str(error)
    else:
        raise AssertionError("missing control encoder weight was accepted")
