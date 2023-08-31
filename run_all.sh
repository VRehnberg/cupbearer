#!/usr/bin/env bash
set -e

root_dir="../cupbearer-experiments/"
mkdir -p $root_dir
cd "$root_dir"

outputfile="$(realpath "results_$(date +%Y-%m-%d_%H-%M-%S).csv")"
echo "Model,Dataset,Backdoor,Detector,AUC_ROC,AP" >> "${outputfile}"

models=( "mlp" "cnn" )
datasets=( "mnist" "cifar10" )
backdoors=( "noise" "corner" "wanet" )
detectors=( "mahalanobis" "abstraction" )

# Backdoors
for model in "${models[@]}"; do

    for dataset in "${datasets[@]}"; do

        for backdoor in "${backdoors[@]}"; do

            task_dir="${model}/${dataset}/${backdoor}/$(date +%Y-%m-%d_%H-%M-%S)/"
            mkdir -p "$task_dir"

            # Train classifier
            python -m cupbearer.scripts.train_classifier \
                --dir.full "$task_dir" \
                --train_data "backdoor" \
                --train_data.original "$dataset" \
                --train_data.backdoor "$backdoor" \
                --train_data.backdoor.p_backdoor "0.1" \
                --model "$model"

            for detector in "${detectors[@]}"; do

                # Train detector
                detector_dir="$task_dir"/"$detector"/"$(date +%Y-%m-%d_%H-%M-%S)"/
                mkdir -p "$detector_dir"

                python -m cupbearer.scripts.train_detector \
                    --dir.full "$detector_dir" \
                    --task backdoor \
                    --task.backdoor "$backdoor" \
                    --task.run_path "$task_dir" \
                    --detector "$detector"

                # Eval detector
                eval_dir="${detector_dir%/}-eval"
                python -m cupbearer.scripts.eval_detector \
                    --save_config \
                    --dir.full "${eval_dir}" \
                    --task backdoor \
                    --task.backdoor "$backdoor" \
                    --task.run_path "$task_dir" \
                    --detector "from_run" \
                    --detector.path "$detector_dir"

                auc_roc=$(jq -r ".AUC_ROC" "${eval_dir}/eval.json")
                ap=$(jq -r ".AP" "${eval_dir}/eval.json")
                echo "${model},${dataset},${backdoor},${detector},${auc_roc},${ap}" >> "${outputfile}"

            done
        done
    done
done
