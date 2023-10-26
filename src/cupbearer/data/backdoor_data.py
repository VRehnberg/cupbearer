# This needs to be in a separate file from backdoors.py because of circularity issues
# with the config groups. See __init__.py.
from dataclasses import dataclass

from torchdata.datapipes.iter import IterableWrapper, IterDataPipe

from cupbearer.data import DatasetConfig
from cupbearer.data.backdoors import Backdoor
from cupbearer.utils.config_groups import config_group


@dataclass
class BackdoorData(DatasetConfig):
    original: DatasetConfig = config_group(DatasetConfig)
    backdoor: Backdoor = config_group(Backdoor)

    @property
    def num_classes(self):
        return self.original.num_classes

    def clean_build(self) -> IterDataPipe:
        dp = IterableWrapper(self.original.build())
        dp = dp.shuffle()  # enable shuffling in DataLoader
        dp = dp.sharding_filter()  # enable sharding for num_workers>1
        dp = dp.map(self.original.transform)
        return dp

    def build(self) -> IterDataPipe:
        dp = self.clean_build()
        dp = dp.map(self.backdoor)
        return dp
