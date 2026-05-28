from odontocal.entry.evaluate import main as evaluate_main
from odontocal.entry.export_onnx import main as export_onnx_main
from odontocal.entry.fit_calibration import main as fit_calibration_main
from odontocal.entry.infer import main as infer_main
from odontocal.entry.train import main as train_main

__all__ = [
    "evaluate_main",
    "export_onnx_main",
    "fit_calibration_main",
    "infer_main",
    "train_main",
]
