from dataclasses import dataclass

from cupbearer.data._shared import DatasetConfig
from cupbearer.data.validation_config import ValidationConfig
from cupbearer.utils.config_groups import config_group
from cupbearer.utils.utils import BaseConfig


@dataclass(kw_only=True)
class DataGroupConfig(BaseConfig):
    train: DatasetConfig = config_group(DatasetConfig)
    val: DatasetConfig = config_group(ValidationConfig, ValidationConfig)

    def __post_init__(self):
        # Make sure to use same backdoor for all datasets in the group
        if hasattr(self.train, "backdoor"):
            for name, val_config in self.val.items():
                # WanetBackdoor
                try:
                    str_factor = (
                        val_config.backdoor.warping_strength
                        / self.train.backdoor.warping_strength
                    )
                    val_config.backdoor.control_grid = (
                        str_factor * self.train.backdoor.control_grid
                    )
                except AttributeError:
                    pass
