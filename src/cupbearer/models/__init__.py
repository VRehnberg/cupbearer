# ruff: noqa: F401
from pathlib import Path

import torch

from .hooked_model import HookedModel
from .models import CNN, MLP, PreActResNet
from .transformers_hf import TamperingPredictionTransformer


def load(model: HookedModel, path: Path | str):
    path = Path(path)
    # Our convention is that LightningModules store the actual pytorch model
    # as a `model` attribute. We use the last checkpoint (generated via the
    # save_last=True option to the ModelCheckpoint callback).
    state_dict = torch.load(
        path / "checkpoints" / "last.ckpt",
        map_location="cpu",  # avoids VRAM surge and is more compatible
    )["state_dict"]
    # We want the state_dict for the 'model' submodule, so remove
    # the 'model.' prefix from the keys.
    state_dict = {k[6:]: v for k, v in state_dict.items() if k.startswith("model.")}
    model.load_state_dict(state_dict)
