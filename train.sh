#!/usr/bin/env bash
set -e

root_dir="../cupbearer-experiments/"
mkdir -p $root_dir
cd "$root_dir"

model="mlp"
dataset="mnist"
backdoor="wanet"

# Train classifier
for lr in $(python -c "import numpy as np; print( *( 10 ** (-4 * np.random.rand(20) ) ) )"); do
    ts_model="$(date +%Y-%m-%d_%H-%M-%S)"
    task_dir="${model}/${dataset}/${backdoor}/${ts_model}/"
    # TODO second might not be unique if run in parallel
    mkdir -p "$task_dir"
    python -m cupbearer.scripts.train_classifier \
        --wandb \
        --dir.full "$task_dir" \
        --train_data "backdoor" \
        --train_data.original "$dataset" \
        --train_data.backdoor "$backdoor" \
        --train_data.backdoor.p_backdoor "0.1" \
        --val_data "backdoor" \
        --val_data.backdoor "backdoor" \
        --val_data.backdoor.backdoor $backdoor \
        --val_data.backdoor.original $dataset \
        --val_data.backdoor.backdoor.p_backdoor "0.1" \
        --val_data.original "backdoor" \
        --val_data.original.backdoor $backdoor \
        --val_data.original.original $dataset \
        --val_data.original.backdoor.p_backdoor "0.0" \
        --val_data.fiftyfifty "backdoor" \
        --val_data.fiftyfifty.backdoor $backdoor \
        --val_data.fiftyfifty.original $dataset \
        --val_data.fiftyfifty.backdoor.p_backdoor "0.5" \
        --optim.learning_rate=$lr \
        --num_epochs=100 \
        --model "$model"
done
