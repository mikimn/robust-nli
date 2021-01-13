#!/bin/bash
# Sets the paths to save the results and the trained models.
results_dir="results/bert/dfl_results/"
mkdir -p $results_dir
outputs_dir="temp/debias/bert/mnli_experiments/"
mkdir -p $outputs_dir
# Sets the paths to the downloaded BERT model and the cache dir.
cache_dir="temp/cache_dir/"
bert_dir="bert-base-uncased"
dropouts=(0.2 0.3 0.4)

for dropout in "${dropouts[@]}"; do
  python run_glue.py \
  --cache_dir $cache_dir \
  --do_train \
  --do_eval \
  --task_name mnli \
  --eval_task_names mnli HANS \
  --model_type bert \
  --num_train_epochs 3 \
  --model_name_or_path $bert_dir \
  --output_dir $outputs_dir/bert_focalloss_hans_dropout_${dropout} \
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
  --hans_features \
  --hidden_dropout_prob ${dropout}
done