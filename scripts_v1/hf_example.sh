model_name="meta-llama/Llama-2-7b-chat-hf"
model_pretty_name="hf_example_test"
TEMP=0.7; TOP_P=1.0; MAX_TOKENS=2048; 
batch_size=1;
# gpu="0,1,2,3"; num_gpus=4; 

CACHE_DIR=${HF_HOME:-"default"}
output_dir="result_dirs/alpaca_eval/"


# Data-parallellism
start_gpu=0
num_gpus=1
n_shards=1
shard_size=256
shards_dir="${output_dir}/tmp_${model_pretty_name}"
for ((start = 0, end = (($shard_size)), gpu = $start_gpu; gpu < $n_shards+$start_gpu; start += $shard_size, end += $shard_size, gpu++)); do

    CUDA_VISIBLE_DEVICES=$gpu \
    python src/unified_infer.py \
        --engine hf \
        --start_index $start --end_index $end \
        --data_name alpaca_eval \
        --model_name $model_name \
        --download_dir $CACHE_DIR \
        --tensor_parallel_size $num_gpus \
        --dtype bfloat16 \
        --top_p $TOP_P --temperature $TEMP \
        --batch_size $batch_size --max_tokens $MAX_TOKENS \
        --output_folder $shards_dir/ \
        --overwrite   
    # break 
done 
wait 
python src/merge_results.py $shards_dir/ $model_pretty_name
cp $shards_dir/${model_pretty_name}.json $output_dir/${model_pretty_name}.json