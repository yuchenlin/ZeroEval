# https://github.com/tatsu-lab/alpaca_eval/blob/f5046aeac0a6d5ea6a665bf2bbf7c8898c1d30ed/src/alpaca_eval/models_configs/Qwen1.5-72B-Chat/configs.yaml
model_name="Qwen/Qwen1.5-72B-Chat"
model_pretty_name="Qwen1.5-72B-Chat"
TEMP=0.7; TOP_P=0.8; MAX_TOKENS=2048; 
gpu="0,1,2,3"; num_gpus=4; batch_size=4;

CACHE_DIR=${HF_HOME:-"default"}
output_dir="result_dirs/wild_bench/"

CUDA_VISIBLE_DEVICES=$gpu \
python src/unified_infer.py \
    --data_name wild_bench \
    --model_name $model_name \
    --model_pretty_name $model_pretty_name \
    --download_dir $CACHE_DIR \
    --tensor_parallel_size $num_gpus \
    --dtype bfloat16 \
    --top_p $TOP_P --temperature $TEMP \
    --batch_size $batch_size --max_tokens $MAX_TOKENS \
    --output_folder $output_dir/ \
    --max_model_len 8192 \
    --overwrite  