model_name=$1 # model to test 
# by default use "gpt-4-0125-preview" as gpt_eval_name
gpt_eval_name=${2:-"gpt-4-turbo-2024-04-09"} # evaluator name  # gpt-4-0125-preview
num_shards=${3:-8} # shards 

total_ex=1024


eval_template="evaluation/eval_template.score.v2.md"
eval_folder="eval_results/v2.0522/score.v2/eval=${gpt_eval_name}/"
echo "Evaluating $model_name using $gpt_eval_name with $eval_template"
mkdir -p $eval_folder 



# Decide the shard size dynamically based on $num_shards and the total number 1024
if [ "$num_shards" -eq 1 ]; then
    eval_file="${eval_folder}/${model_name}.json"
    python src/eval.py \
        --action eval \
        --model $gpt_eval_name \
        --max_words_to_eval 1000 \
        --mode score \
        --eval_template $eval_template \
        --target_model_name $model_name \
        --eval_output_file $eval_file 
else
    echo "Using $num_shards shards"
    shard_size=$(($total_ex / $num_shards))
    echo "Shard size: $shard_size"
    start_gpu=0 # not used 
    for ((start = 0, end = (($shard_size)), gpu = $start_gpu; gpu < $num_shards+$start_gpu; start += $shard_size, end += $shard_size, gpu++)); do
        eval_file="${eval_folder}/${model_name}.$start-$end.json"
        echo "Evaluating $model_name vs $ref_name from $start to $end"
        python src/eval.py \
            --action eval \
            --model $gpt_eval_name \
            --max_words_to_eval 1000 \
            --mode score \
            --eval_template $eval_template \
            --target_model_name $model_name \
            --eval_output_file $eval_file \
            --start_idx $start --end_idx $end \
            &
    done 
    # Wait for all background processes to finish
    wait

    # # Run the merge results script after all evaluation scripts have completed
    python src/merge_results.py $eval_folder $model_name
fi
 

# bash evaluation/run_eval_v2.score.sh claude-3-opus-20240229 claude-3-opus-20240229 1
# bash evaluation/run_eval_v2.score.sh gpt-4o-2024-05-13 claude-3-opus-20240229
# bash evaluation/run_eval_v2.score.sh gpt-4-turbo-2024-04-09 claude-3-opus-20240229
# bash evaluation/run_eval_v2.score.sh gpt-3.5-turbo-0125 claude-3-opus-20240229
# bash evaluation/run_eval_v2.score.sh claude-3-haiku-20240307 claude-3-opus-20240229
# bash evaluation/run_eval_v2.score.sh claude-3-sonnet-20240229 claude-3-opus-20240229
# bash evaluation/run_eval_v2.score.sh Meta-Llama-3-8B-Instruct claude-3-opus-20240229
# bash evaluation/run_eval_v2.score.sh Meta-Llama-3-70B-Instruct claude-3-opus-20240229
# bash evaluation/run_eval_v2.score.sh dbrx-instruct@together claude-3-opus-20240229

# bash evaluation/run_eval_v2.score.sh gemma-2b-it claude-3-opus-20240229
# bash evaluation/run_eval_v2.score.sh gemma-7b-it claude-3-opus-20240229
# bash evaluation/run_eval_v2.score.sh Llama-2-7b-chat-hf claude-3-opus-20240229
# bash evaluation/run_eval_v2.score.sh Llama-2-70b-chat-hf claude-3-opus-20240229

# bash evaluation/run_eval_v2.score.sh Mistral-7B-Instruct-v0.2 claude-3-opus-20240229
# bash evaluation/run_eval_v2.score.sh mistral-large-2402 claude-3-opus-20240229
# bash evaluation/run_eval_v2.score.sh Mixtral-8x7B-Instruct-v0.1 claude-3-opus-20240229
# bash evaluation/run_eval_v2.score.sh Nous-Hermes-2-Mixtral-8x7B-DPO claude-3-opus-20240229
# bash evaluation/run_eval_v2.score.sh Qwen1.5-7B-Chat@together claude-3-opus-20240229
# bash evaluation/run_eval_v2.score.sh Qwen1.5-72B-Chat claude-3-opus-20240229
# bash evaluation/run_eval_v2.score.sh Starling-LM-7B-beta claude-3-opus-20240229
# bash evaluation/run_eval_v2.score.sh tulu-2-dpo-70b claude-3-opus-20240229
 

