import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from model import get_model


def test_resnet18_output_shape():
    model = get_model(architecture="resnet18", num_classes=10)
    model.eval()
    dummy_input = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        output = model(dummy_input)
    assert output.shape == (2, 10)


def test_simple_cnn_output_shape():
    model = get_model(architecture="simple_cnn", num_classes=10)
    model.eval()
    dummy_input = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        output = model(dummy_input)
    assert output.shape == (2, 10)


def test_unknown_architecture_raises():
    try:
        get_model(architecture="not_a_real_model", num_classes=10)
        assert False, "expected ValueError"
    except ValueError:
        pass
