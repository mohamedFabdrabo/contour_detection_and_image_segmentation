import importlib.util
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]

spec_ms = importlib.util.spec_from_file_location(
    "local.MySegmentation", ROOT / "code" / "MySegmentation.py"
)
MySegmentation = importlib.util.module_from_spec(spec_ms)
spec_ms.loader.exec_module(MySegmentation)

spec_fcn = importlib.util.spec_from_file_location(
    "local.FCN", ROOT / "code" / "FCN.py"
)
FCN = importlib.util.module_from_spec(spec_fcn)
spec_fcn.loader.exec_module(FCN)


def test_mean_shift_segmentation_import():
    assert hasattr(MySegmentation, "MeanShiftSegmentation")
    assert callable(MySegmentation.MeanShiftSegmentation)


def test_fcn_resnet101_import():
    assert hasattr(FCN, "FCN_RESNET101")
    assert callable(FCN.FCN_RESNET101)
