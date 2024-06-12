# export GOOGLE_API_KEY=your_google_studio_api_key
DATA_NAME=$1
COT=$2
model_name="google/gemini-1.0-pro"
model_pretty_name="gemini-1.0-pro"
output_dir="result_dirs/$DATA_NAME/cot=$COT/"
TEMP=0; TOP_P=1.0; MAX_TOKENS=4096;

# shard_size should be 1024 // n_shards
# if DATA_NAME is "mmlu-redux", then set number of examples as 3k; if it is "gsm", then set it as 1.5k
if [ $DATA_NAME == "mmlu-redux" ]; then
    NUM_EXAMPLES=2800
elif [ $DATA_NAME == "gsm" ]; then
    NUM_EXAMPLES=1500
fi


# shard_size should be 1024 // n_shards
n_shards=8
shard_size=$((NUM_EXAMPLES / n_shards))
start_gpu=0
shards_dir="${output_dir}/tmp_${model_pretty_name}"
for ((start = 0, end = (($shard_size)), gpu = $start_gpu; gpu < $n_shards+$start_gpu; start += $shard_size, end += $shard_size, gpu++)); do
    python src/unified_infer.py \
        --data_name $DATA_NAME --cot $COT \
        --start_index $start --end_index $end \
        --engine google \
        --model_name $model_name \
        --top_p $TOP_P --temperature $TEMP \
        --max_tokens $MAX_TOKENS \
        --output_folder $shards_dir/ \
         &
done 
wait 
python src/merge_results.py $shards_dir/ $model_pretty_name
cp $shards_dir/${model_pretty_name}.json $output_dir/${model_pretty_name}.json
