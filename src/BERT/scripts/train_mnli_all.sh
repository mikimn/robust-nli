#!/bin/bash
seeds=(186 44)
# Sets the paths to save the results and the trained models.
results_dir="results/bert/mnli_results/"
mkdir -p $results_dir
outputs_dir="temp/debias/bert/mnli_experiments/"
mkdir -p $outputs_dir
# Sets the paths to the downloaded BERT model and the cache dir.
cache_dir="temp/cache_dir/"
bert_dir="bert-base-uncased"


# Train the baseline model.
for seed in "${seeds[@]}"; do
  python run_glue.py \
  --cache_dir $cache_dir \
  --do_train \
  --do_eval \
  --task_name mnli \
  --eval_task_names all \
  --model_type bert \
  --num_train_epochs 3 \
  --lambda_h 0.0 \
  --model_name_or_path $bert_dir \
  --output_dir $outputs_dir/bert_baseline_seed_"$seed" \
  --overwrite_output_dir \
  --learning_rate 2e-5 \
  --do_lower_case \
  --outputfile $results_dir/results.csv \
  --binerize_eval \
  --seed "$seed"
done

# Train the DFL model.
#for seed in "${seeds[@]}"; do
#  python run_glue.py \
#  --cache_dir $cache_dir \
#  --do_eval \
#  --do_train \
#  --task_name mnli \
#  --eval_task_names all \
#  --model_type bert \
#  --num_train_epochs 3 \
#  --lambda_h 1.0  \
#  --model_name_or_path $bert_dir \
#  --output_dir $outputs_dir/bert_focalloss_seed_"$seed" \
#  --overwrite_output_dir \
#  --learning_rate 2e-5 \
#  --do_lower_case \
#  --outputfile $results_dir/results.csv \
#  --focal_loss \
#  --seed "$seed" \
#  --nonlinear_h_classifier deep \
#  --binerize_eval
#done

# Train the POE model.
#python run_glue.py --cache_dir $cache_dir --do_train  --do_eval \
#--task_name mnli  --eval_task_names  all --model_type bert --num_train_epochs 3 \
#--lambda_h 1.0  --model_name_or_path $bert_dir --output_dir $outputs_dir/bert_poeloss \
#--overwrite_output_dir --learning_rate 2e-5 --do_lower_case --outputfile $results_dir/results.csv \
#--poe_loss   --nonlinear_h_classifier deep  --binerize_eval
