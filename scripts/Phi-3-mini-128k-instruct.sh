DATA_NAME=$1
COT=$2
model_name="microsoft/Phi-3-mini-128k-instruct"
model_pretty_name="Phi-3-mini-128k-instruct"
TEMP=0; TOP_P=1.0; MAX_TOKENS=4096; 
batch_size=4;
# gpu="0,1,2,3"; num_gpus=4; 

CACHE_DIR=${HF_HOME:-"default"}
output_dir="result_dirs/$DATA_NAME/cot=$COT/"



# if DATA_NAME is "mmlu-redux", then set number of examples as 3k; if it is "gsm", then set it as 1.5k
if [ $DATA_NAME == "mmlu-redux" ]; then
    NUM_EXAMPLES=2800
elif [ $DATA_NAME == "gsm" ]; then
    NUM_EXAMPLES=1500
fi

# Data-parallellism
start_gpu=0
num_gpus=1


# shard_size should be 1024 // n_shards
n_shards=4
shard_size=$((NUM_EXAMPLES / n_shards))
shards_dir="${output_dir}/tmp_${model_pretty_name}"
for ((start = 0, end = (($shard_size)), gpu = $start_gpu; gpu < $n_shards+$start_gpu; start += $shard_size, end += $shard_size, gpu++)); do

    CUDA_VISIBLE_DEVICES=$gpu \
    python src/unified_infer.py \
        --start_index $start --end_index $end \
        --data_name $DATA_NAME --cot $COT \
        --model_name $model_name \
        --max_model_len 40960 \
        --use_hf_conv_template \
        --download_dir $CACHE_DIR \
        --tensor_parallel_size $num_gpus \
        --dtype bfloat16 \
        --model_pretty_name $model_pretty_name \
        --top_p $TOP_P --temperature $TEMP \
        --batch_size $batch_size --max_tokens $MAX_TOKENS \
        --output_folder $shards_dir/ \
        --overwrite  &
done 
wait 
python src/merge_results.py $shards_dir/ $model_pretty_name
cp $shards_dir/${model_pretty_name}.json $output_dir/${model_pretty_name}.json