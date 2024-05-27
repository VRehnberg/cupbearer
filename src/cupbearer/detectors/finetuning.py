import copy
import types
import warnings
from pathlib import Path
from typing import Callable

import lightning as L
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from cupbearer.detectors.anomaly_detector import AnomalyDetector
from cupbearer.scripts._shared import Classifier
from cupbearer.utils import inputs_from_batch


class VerboseCallback(L.pytorch.callbacks.Callback):
    def __init__(self, verbose: bool = True):
        super().__init__()
        self.verbose = verbose

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        super().on_before_optimizer_step(trainer, pl_module, optimizer)
        if self.verbose:
            print(f"Taking optimizer step with opt={optimizer}")


class FinetuningAnomalyDetector(AnomalyDetector):
    def set_model(self, model):
        super().set_model(model)
        # We might as well make a copy here already, since whether we'll train this
        # detector or load weights for inference, we'll need to copy in both cases.
        self.finetuned_model = copy.deepcopy(self.model)

    def train(
        self,
        trusted_data,
        untrusted_data,
        save_path: Path | str,
        *,
        num_classes: int,
        lr: float = 1e-3,
        batch_size: int = 64,
        num_workers=0,
        verbose: bool = False,
        configure_optimizers: Callable = None,
        **trainer_kwargs,
    ):
        if trusted_data is None:
            raise ValueError("Finetuning detector requires trusted training data.")
        classifier = Classifier(
            self.finetuned_model,
            num_classes=num_classes,
            lr=lr,
            save_hparams=False,
        )
        if configure_optimizers is not None:
            if verbose:
                print('Setting "configure_optimizers"')
            # For flexibility allow monkey patching in optimizers
            classifier.configure_optimizers = types.MethodType(
                configure_optimizers, classifier
            )

        # Create a DataLoader for the clean dataset
        clean_loader = DataLoader(
            trusted_data, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )

        # Finetune the model on the clean dataset
        trainer = L.Trainer(
            default_root_dir=save_path,
            callbacks=VerboseCallback(verbose=verbose),
            **trainer_kwargs,
        )
        if verbose:
            print(f"Trainer initialized with opt={trainer.optimizers}")

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=(
                    "You defined a `validation_step` but have no `val_dataloader`."
                    " Skipping val loop."
                ),
            )
            trainer.fit(
                model=classifier,
                train_dataloaders=clean_loader,
            )

    def layerwise_scores(self, batch):
        raise NotImplementedError(
            "Layerwise scores don't exist for finetuning detector"
        )

    def scores(self, batch):
        inputs = inputs_from_batch(batch)
        original_output = self.model(inputs)
        finetuned_output = self.finetuned_model(inputs)

        # F.kl_div requires log probabilities for the input, normal probabilities
        # are fine for the target.
        log_finetuned_p = finetuned_output.log_softmax(dim=-1)
        original_p = original_output.softmax(dim=-1)

        # This computes KL(original || finetuned), the argument order for the pytorch
        # function is swapped compared to the mathematical notation.
        # See https://pytorch.org/docs/stable/generated/torch.nn.KLDivLoss.html
        # This is the same direction of KL divergence that Redwood used in one of their
        # projects, though I don't know if they had a strong reason for it.
        # Arguably a symmetric metric would make more sense, but might not matter much.
        #
        # Also note we don't want pytorch to do any reduction, since we want to
        # return individual scores for each sample.
        kl = F.kl_div(log_finetuned_p, original_p, reduction="none").sum(-1)

        if torch.any(torch.isinf(kl)):
            # We'd get an error anyway once we compute eval metrics, but better to give
            # a more specific one here.
            raise ValueError("Infinite KL divergence")

        return kl

    def _get_trained_variables(self, saving: bool = False):
        return self.finetuned_model.state_dict()

    def _set_trained_variables(self, variables):
        self.finetuned_model.load_state_dict(variables)
