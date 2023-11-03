from dataclasses import dataclass, field

from cupbearer.detectors.config import ActivationBasedDetectorConfig, TrainConfig

from .mahalanobis_detector import MahalanobisDetector


@dataclass
class MahalanobisTrainConfig(TrainConfig):
    max_batches: int = 0
    relative: bool = False
    rcond: float = 1e-5
    batch_size: int = 4096
    pbar: bool = True
    debug: bool = False

    def setup_and_validate(self):
        super().setup_and_validate()
        if self.debug:
            self.max_batches = 2
            self.batch_size = 2


@dataclass
class MahalanobisConfig(ActivationBasedDetectorConfig):
    train: MahalanobisTrainConfig = field(default_factory=MahalanobisTrainConfig)

    def build(self, model, save_dir) -> MahalanobisDetector:
        return MahalanobisDetector(
            model=model,
            activation_names=self.get_names(model),
            max_batch_size=self.max_batch_size,
            save_path=save_dir,
        )
