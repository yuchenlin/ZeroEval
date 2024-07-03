DATA_NAME=$1
engine_name=$2
model_name=$3
model_pretty_name=$4
n_shards=$5
# default cot to be True 
cot=${6:-True}

TEMP=0; TOP_P=1.0; MAX_TOKENS=4096; 
batch_size=4; 
CACHE_DIR=${HF_HOME:-"default"}
output_dir="result_dirs/${DATA_NAME}/cot=${cot}/" 

# If the n_shards is 1, then we can directly run the model
# else, use  Data-parallellism
if [ $n_shards -eq 1 ]; then
    echo "n_shards = 1"
    CUDA_VISIBLE_DEVICES=$gpu \
    python src/unified_infer.py \
        --data_name $DATA_NAME \
        --cot $cot \
        --engine $engine_name \
        --model_name $model_name \
        --top_p $TOP_P --temperature $TEMP \
        --batch_size $batch_size --max_tokens $MAX_TOKENS \
        --output_folder $output_dir/  

elif [ $n_shards -gt 1 ]; then
    echo "Using Data-parallelism"
    start_gpu=0 
    shards_dir="${output_dir}/tmp_${model_pretty_name}"
    for ((shard_id = 0; shard_id < $n_shards; shard_id++, gpu++)); do
        python src/unified_infer.py \
            --num_shards $n_shards \
            --shard_id $shard_id \
            --data_name $DATA_NAME \
            --cot $cot \
            --engine $engine_name \
            --model_name $model_name \
            --model_pretty_name $model_pretty_name \
            --top_p $TOP_P --temperature $TEMP \
            --batch_size $batch_size --max_tokens $MAX_TOKENS \
            --output_folder $shards_dir/ \
              &
    done 
    wait 
    python src/merge_results.py $shards_dir/ $model_pretty_name
    cp $shards_dir/${model_pretty_name}.json $output_dir/${model_pretty_name}.json
else
    echo "Invalid n_shards"
    exit
fi

