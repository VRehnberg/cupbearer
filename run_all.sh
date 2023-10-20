#!/usr/bin/env bash
set -e

root_dir="../cupbearer-experiments/"
mkdir -p $root_dir
cd "$root_dir"

#outputfile="$(realpath "results_$(date +%Y-%m-%d_%H-%M-%S).csv")"
#echo "Model,Dataset,Backdoor,Detector,Train acc.,AUC_ROC,AP,Timestamp Model,Timestamp Detector" >> "${outputfile}"

models=( "mlp" "cnn" )
datasets=( "mnist" "cifar10" )
backdoors=( "noise" "corner" "wanet" )
detectors=( "mahalanobis" "abstraction" )

# Backdoors
for model in "${models[@]}"; do

    for dataset in "${datasets[@]}"; do

        for backdoor in "${backdoors[@]}"; do

            ts_model="$(date +%Y-%m-%d_%H-%M-%S)"
            task_dir="${model}/${dataset}/${backdoor}/${ts_model}/"
            # TODO second might not be unique if run in parallel
            mkdir -p "$task_dir"

            # Train classifier
            python -m cupbearer.scripts.train_classifier \
                --wandb \
                --dir.full "$task_dir" \
                --train_data "backdoor" \
                --train_data.original "$dataset" \
                --train_data.backdoor "$backdoor" \
                --train_data.backdoor.p_backdoor "0.1" \
                --model "$model"

            #train_acc=$(jq -r '.["10"]["train/accuracy"]' "${task_dir}/metrics.json")

            for detector in "${detectors[@]}"; do

                # Train detector
                ts_detector="$(date +%Y-%m-%d_%H-%M-%S)"
                detector_dir="${task_dir}/${detector}/${ts_detector}/"
                mkdir -p "$detector_dir"

                python -m cupbearer.scripts.train_detector \
                    --wandb \
                    --dir.full "$detector_dir" \
                    --task backdoor \
                    --task.backdoor "$backdoor" \
                    --task.run_path "$task_dir" \
                    --detector "$detector"

                ## Eval detector
                #eval_dir="${detector_dir%/}-eval"
                #python -m cupbearer.scripts.eval_detector \
                #    --wandb \
                #    --save_config \
                #    --dir.full "${eval_dir}" \
                #    --task backdoor \
                #    --task.backdoor "$backdoor" \
                #    --task.run_path "$task_dir" \
                #    --detector "from_run" \
                #    --detector.path "$detector_dir"

                ##auc_roc=$(jq -r ".AUC_ROC" "${eval_dir}/eval.json")
                ##ap=$(jq -r ".AP" "${eval_dir}/eval.json")
                ##echo "${model},${dataset},${backdoor},${detector},${train_acc},${auc_roc},${ap},${ts_model},${ts_detector}" >> "${outputfile}"

                break #TODO tmp

            done
            break #TODO tmp
        done
        break #TODO tmp
    done
    break #TODO tmp
done
