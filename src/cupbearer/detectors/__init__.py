from cupbearer.utils.config_groups import register_config_group

from .abstraction import AbstractionDetectorConfig
from .config import DetectorConfig, StoredDetector
from .finetuning import FinetuningConfig
from .mahalanobis import MahalanobisConfig

DETECTORS = {
    "abstraction": AbstractionDetectorConfig,
    # "adversarial_abstraction": AdversarialAbstractionConfig,
    "mahalanobis": MahalanobisConfig,
    "finetuning": FinetuningConfig,
    "from_run": StoredDetector,
}

register_config_group(DetectorConfig, DETECTORS)
