#!/bin/bash
# Sets the paths to save the results and the trained models.
results_dir="results/bert/dfl_results/"
mkdir -p $results_dir
outputs_dir="temp/debias/bert/mnli_experiments/"
mkdir -p $outputs_dir
# Sets the paths to the downloaded BERT model and the cache dir.
cache_dir="temp/cache_dir/"
bert_dir="bert-base-uncased"
gammas=(1.0 2.0 3.0 4.0)

# Train the DFL model.
for gamma in "${gammas[@]}"; do
  python run_glue.py \
  --cache_dir $cache_dir  \
  --do_eval \
  --do_train \
  --task_name mnli \
  --eval_task_names all \
  --model_type bert \
  --num_train_epochs 3 \
  --gamma_focal ${gamma} \
  --model_name_or_path $bert_dir \
  --output_dir $outputs_dir/bert_focalloss_mnli_gamma_${gamma} \
  --overwrite_output_dir \
  --learning_rate 2e-5 \
  --do_lower_case \
  --outputfile $results_dir/results_lambda.csv \
  --focal_loss \
  --nonlinear_h_classifier deep \
  --binerize_eval
done

for gamma in "${gammas[@]}"; do
  python run_glue.py \
  --cache_dir $cache_dir \
  --do_train \
  --do_eval \
  --task_name mnli \
  --eval_task_names mnli HANS \
  --model_type bert \
  --num_train_epochs 3 \
  --gamma_focal ${gamma} \
  --model_name_or_path $bert_dir \
  --output_dir $outputs_dir/bert_focalloss_hans_gamma_${gamma} \
  --overwrite_output_dir \
  --learning_rate 2e-5 \
  --do_lower_case \
  --outputfile $results_dir/results_lambda.csv \
  --focal_loss \
  --nonlinear_h_classifier deep \
  --hans \
  --similarity mean max min \
  --weighted_bias_only \
  --binerize_eval \
  --seed $seed \
  --hans_features
done