#!/usr/bin/env bash
set -e

root_dir="../cupbearer-experiments/"
mkdir -p $root_dir
cd "$root_dir"

N=100
model="cnn"
dataset="cifar10"
backdoor="wanet"

unset is_wanet
if [ "$backdoor" = "wanet" ]; then
    is_wanet=""
fi

# Train classifier
for lr in $(python -c "import numpy as np; print(*( 10 ** (np.random.uniform(-4, -2, ${N:-"1"})) ))"); do
    ts_model="$(date +%Y-%m-%d_%H-%M-%S)"
    task_dir="${model}/${dataset}/${backdoor}/${ts_model}_${SLURM_JOBID:-"$BASHPID"}/"
    mkdir -p "$task_dir"
    python -m cupbearer.scripts.train_classifier \
        --wandb \
        --dir.full "$task_dir" \
        --train_data "backdoor" \
        --train_data.original "$dataset" \
        --train_data.backdoor "$backdoor" \
        --train_data.backdoor.p_backdoor 0.1 \
        --val_data.val "backdoor" \
        --val_data.val.original "$dataset" \
        --val_data.val.original.train False \
        --val_data.val.backdoor "$backdoor" \
        --val_data.val.backdoor.p_backdoor "0.1" \
        ${is_wanet+"--val_data.val.backdoor.p_noise=0.0"} \
        --val_data.clean "backdoor" \
        --val_data.clean.original "$dataset" \
        --val_data.clean.original.train False \
        --val_data.clean.backdoor "$backdoor" \
        --val_data.clean.backdoor.p_backdoor "0.0" \
        ${is_wanet+"--val_data.clean.backdoor.p_noise=0.0"} \
        --val_data.backdoor "backdoor" \
        --val_data.backdoor.original "$dataset" \
        --val_data.backdoor.original.train False \
        --val_data.backdoor.backdoor "$backdoor" \
        ${is_wanet+"--val_data.backdoor.backdoor.p_noise=0.0"} \
        --val_data.custom "backdoor" \
        --val_data.custom.original "$dataset" \
        --val_data.custom.original.train False \
        --val_data.custom.backdoor "$backdoor" \
        --val_data.custom.backdoor.p_backdoor "0.0" \
        ${is_wanet+"--val_data.custom.backdoor.p_noise=1.0"} \
        --optim.learning_rate=$lr \
        --num_epochs=100 \
        ${is_wanet-"--num_workers=$((SLURM_CPUS_ON_NODE / SLURM_GPUS_ON_NODE - 1))"} \
        --model "$model"
done
