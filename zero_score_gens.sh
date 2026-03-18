#!/bin/bash
# Usage:  bash zero_score_gens.sh -d $TASK_NAME

DATA_NAME="unk"

# Parse named arguments
while getopts ":d:" opt; do
  case $opt in
    d) DATA_NAME="$OPTARG"
    ;;
    \?) echo "Invalid option -$OPTARG" >&2
    ;;
  esac
done


if [ $DATA_NAME == "all" ]; then
    python -m src.evaluation.summarize  

elif [ $DATA_NAME == "zebra_grid" ]; then
    python -m src.evaluation.zebra_grid_eval    

elif [ $DATA_NAME == "hendrycks-math" ]; then
    python -m src.evaluation.hendrycks_math_eval

elif [ $DATA_NAME == "mmlu-pro" ]; then
    python -m src.evaluation.mmlu_pro_eval mmlu-pro

elif [ $DATA_NAME == "mmlu-pro-short" ]; then
    python -m src.evaluation.mmlu_pro_eval mmlu-pro-short

elif [ $DATA_NAME == "crux" ]; then
    python -m src.evaluation.crux_eval

elif [ $DATA_NAME == "math-l5" ]; then
    python -m src.evaluation.math_eval math-l5

elif [ $DATA_NAME == "gsm" ]; then
    python -m src.evaluation.math_eval gsm

elif [ $DATA_NAME == "mmlu-redux" ]; then
    python -m src.evaluation.mcqa_eval mmlu-redux

elif [ $DATA_NAME == "gplanet" ]; then
    python -m src.evaluation.gplanet_eval

else
    echo "Can't get accuracy scores for dataset. Invalid dataset_name: '$DATA_NAME'"
    exit 1
fi

echo "All evaluations completed!"