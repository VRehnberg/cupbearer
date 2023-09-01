#!/usr/bin/env bash
set -e

root_dir="../cupbearer-experiments/"
mkdir -p $root_dir
cd "$root_dir"

<<<<<<< HEAD
#outputfile="$(realpath "results_$(date +%Y-%m-%d_%H-%M-%S).csv")"
#echo "Model,Dataset,Backdoor,Detector,Train acc.,AUC_ROC,AP,Timestamp Model,Timestamp Detector" >> "${outputfile}"
=======
outputfile="$(realpath "results_$(date +%Y-%m-%d_%H-%M-%S).csv")"
echo "Model,Dataset,Backdoor,Detector,Train acc.,AUC_ROC,AP" >> "${outputfile}"
>>>>>>> af08d61 (Add train acc. to results csv)

models=( "mlp" "cnn" )
datasets=( "mnist" "cifar10" )
backdoors=( "noise" "corner" "wanet" )
detectors=( "mahalanobis" "abstraction" )

# Backdoors
for model in "${models[@]}"; do

    for dataset in "${datasets[@]}"; do

        for backdoor in "${backdoors[@]}"; do

<<<<<<< HEAD
            ts_model="$(date +%Y-%m-%d_%H-%M-%S)"
            task_dir="${model}/${dataset}/${backdoor}/${ts_model}/"
=======
            task_dir="${model}/${dataset}/${backdoor}/$(date +%Y-%m-%d_%H-%M-%S)/"
>>>>>>> af08d61 (Add train acc. to results csv)
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

<<<<<<< HEAD
            #train_acc=$(jq -r '.["10"]["train/accuracy"]' "${task_dir}/metrics.json")
=======
            train_acc=$(jq -r '.["10"]["train/accuracy"]' "${task_dir}/metrics.json")
>>>>>>> af08d61 (Add train acc. to results csv)

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

<<<<<<< HEAD
                ##auc_roc=$(jq -r ".AUC_ROC" "${eval_dir}/eval.json")
                ##ap=$(jq -r ".AP" "${eval_dir}/eval.json")
                ##echo "${model},${dataset},${backdoor},${detector},${train_acc},${auc_roc},${ap},${ts_model},${ts_detector}" >> "${outputfile}"

                break #TODO tmp
=======
                auc_roc=$(jq -r ".AUC_ROC" "${eval_dir}/eval.json")
                ap=$(jq -r ".AP" "${eval_dir}/eval.json")
                echo "${model},${dataset},${backdoor},${detector},${train_acc},${auc_roc},${ap}" >> "${outputfile}"
>>>>>>> af08d61 (Add train acc. to results csv)

            done
            break #TODO tmp
        done
        break #TODO tmp
    done
    break #TODO tmp
done
