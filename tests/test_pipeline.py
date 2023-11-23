import pytest
import torch
from cupbearer.scripts import (
    eval_classifier,
    train_classifier,
    train_detector,
)
from cupbearer.scripts.conf import (
    eval_classifier_conf,
    train_classifier_conf,
    train_detector_conf,
)
from cupbearer.utils.scripts import run
from simple_parsing import ArgumentGenerationMode, parse

# Ignore warnings about num_workers
pytestmark = pytest.mark.filterwarnings(
    "ignore"
    ":The '[a-z]*_dataloader' does not have many workers which may be a bottleneck. "
    "Consider increasing the value of the `num_workers` argument` to "
    "`num_workers=[0-9]*` in the `DataLoader` to improve performance."
    ":UserWarning"
)


@pytest.fixture(scope="module")
def backdoor_classifier_path(module_tmp_path):
    """Trains a backdoored classifier and returns the path to the run directory."""
    cfg = parse(
        train_classifier_conf.Config,
        args=f"--debug_with_logging --dir.full {module_tmp_path} "
        "--train_data backdoor --train_data.original mnist "
        "--train_data.backdoor corner --model mlp",
        argument_generation_mode=ArgumentGenerationMode.NESTED,
    )
    run(train_classifier.main, cfg)

    assert (module_tmp_path / "config.yaml").is_file()
    assert (module_tmp_path / "checkpoints" / "last.ckpt").is_file()
    assert (module_tmp_path / "tensorboard").is_dir()

    return module_tmp_path


@pytest.mark.slow
def test_eval_classifier(backdoor_classifier_path):
    cfg = parse(
        eval_classifier_conf.Config,
        args=f"--debug_with_logging --dir.full {backdoor_classifier_path} "
        "--data mnist --data.train false",
        argument_generation_mode=ArgumentGenerationMode.NESTED,
    )
    run(eval_classifier.main, cfg)

    assert (backdoor_classifier_path / "eval.json").is_file()


@pytest.mark.slow
def test_train_abstraction_corner_backdoor(backdoor_classifier_path, tmp_path):
    cfg = parse(
        train_detector_conf.Config,
        args=f"--debug_with_logging --dir.full {tmp_path} "
        f"--task backdoor --task.backdoor corner "
        f"--task.path {backdoor_classifier_path} --detector abstraction",
        argument_generation_mode=ArgumentGenerationMode.NESTED,
    )
    run(train_detector.main, cfg)
    assert (tmp_path / "config.yaml").is_file()
    assert (tmp_path / "detector.pt").is_file()

    assert (tmp_path / "histogram.pdf").is_file()
    assert (tmp_path / "eval.json").is_file()


@pytest.mark.slow
def test_train_autoencoder_corner_backdoor(backdoor_classifier_path, tmp_path):
    cfg = parse(
        train_detector_conf.Config,
        args=f"--debug_with_logging --dir.full {tmp_path} "
        f"--task backdoor --task.backdoor corner "
        f"--task.path {backdoor_classifier_path} --detector abstraction "
        f"--detector.abstraction autoencoder",
        argument_generation_mode=ArgumentGenerationMode.NESTED,
    )
    run(train_detector.main, cfg)
    assert (tmp_path / "config.yaml").is_file()
    assert (tmp_path / "detector.pt").is_file()

    assert (tmp_path / "histogram.pdf").is_file()
    assert (tmp_path / "eval.json").is_file()


def filter_advex_failure(err, *args):
    flaky_error_msg = (
        "RuntimeError(\nRuntimeError: Attack failed, new accuracy is 100.0 > 1.0.\n"
    )
    return err[1].stderr.endswith(flaky_error_msg)


@pytest.mark.flaky(max_runs=3, rerun_filter=filter_advex_failure)
@pytest.mark.slow
def test_train_mahalanobis_advex(backdoor_classifier_path, tmp_path):
    # This test doesn't need a backdoored classifier, but we already have one
    # and it doesn't hurt, so reusing it makes execution faster.
    cfg = parse(
        train_detector_conf.Config,
        args=f"--debug_with_logging --dir.full {tmp_path} "
        f"--task adversarial_examples --task.path {backdoor_classifier_path} "
        "--detector mahalanobis",
        argument_generation_mode=ArgumentGenerationMode.NESTED,
    )
    run(train_detector.main, cfg)
    assert (backdoor_classifier_path / "adv_examples.pt").is_file()
    assert (backdoor_classifier_path / "adv_examples.pdf").is_file()
    assert (tmp_path / "config.yaml").is_file()
    assert (tmp_path / "detector.pt").is_file()
    # Eval outputs:
    assert (tmp_path / "histogram.pdf").is_file()
    assert (tmp_path / "eval.json").is_file()


@pytest.mark.slow
def test_train_mahalanobis_backdoor(backdoor_classifier_path, tmp_path):
    cfg = parse(
        train_detector_conf.Config,
        args=f"--debug_with_logging --dir.full {tmp_path} "
        f"--task backdoor --task.backdoor corner "
        f"--task.path {backdoor_classifier_path} --detector mahalanobis",
        argument_generation_mode=ArgumentGenerationMode.NESTED,
    )
    run(train_detector.main, cfg)
    assert (tmp_path / "config.yaml").is_file()
    assert (tmp_path / "detector.pt").is_file()
    # Eval outputs:
    assert (tmp_path / "histogram.pdf").is_file()
    assert (tmp_path / "eval.json").is_file()


@pytest.mark.slow
def test_finetuning_detector(backdoor_classifier_path, tmp_path):
    cfg = parse(
        train_detector_conf.Config,
        args=f"--debug_with_logging --dir.full {tmp_path} "
        f"--task backdoor --task.backdoor corner "
        f"--task.path {backdoor_classifier_path} --detector finetuning",
        argument_generation_mode=ArgumentGenerationMode.NESTED,
    )
    run(train_detector.main, cfg)
    assert (tmp_path / "config.yaml").is_file()
    assert (tmp_path / "detector.pt").is_file()

    assert (tmp_path / "histogram.pdf").is_file()
    assert (tmp_path / "eval.json").is_file()


@pytest.mark.slow
def test_wanet(tmp_path):
    cfg = parse(
        train_classifier_conf.Config,
        args=f"--debug_with_logging --dir.full {tmp_path / 'wanet'} "
        "--train_data backdoor --train_data.original gtsrb "
        "--train_data.backdoor wanet --model mlp "
        "--val_data.backdoor backdoor --val_data.backdoor.original gtsrb "
        "--val_data.backdoor.backdoor wanet "
        "--num_workers=1",
        argument_generation_mode=ArgumentGenerationMode.NESTED,
    )
    run(train_classifier.main, cfg)

    assert (tmp_path / "wanet" / "config.yaml").is_file()
    assert (tmp_path / "wanet" / "checkpoints" / "last.ckpt").is_file()
    assert (tmp_path / "wanet" / "tensorboard").is_dir()
    # Check that NoData is handled correctly
    for name, data_cfg in cfg.val_data.items():
        if name == "backdoor":
            assert torch.allclose(
                data_cfg.backdoor.control_grid,
                cfg.train_data.backdoor.control_grid,
            )
        else:
            with pytest.raises(NotImplementedError):
                data_cfg.build()

    # Check that from_run can load WanetBackdoor properly
    train_detector_cfg = parse(
        train_detector_conf.Config,
        args=f"--debug_with_logging --dir.full {tmp_path / 'wanet-mahalanobis'} "
        f"--task backdoor --task.backdoor wanet --task.path {tmp_path / 'wanet'} "
        "--detector mahalanobis",
        argument_generation_mode=ArgumentGenerationMode.NESTED,
    )
    run(train_detector.main, train_detector_cfg)
    assert torch.allclose(
        train_detector_cfg.task.backdoor.control_grid,
        cfg.train_data.backdoor.control_grid,
    )
