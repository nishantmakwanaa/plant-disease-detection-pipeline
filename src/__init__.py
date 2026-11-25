"""
Plant Disease Detection Pipeline — package init.
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("plant-disease-detection-pipeline")
except PackageNotFoundError:
    __version__ = "0.1.0"

__all__ = ["model", "dataset", "train", "evaluate", "inference", "gradcam"]
