model_name=$1 # model to test   
gpt_eval_name=${2:-"gpt-4-0125-preview"} # evaluator name  # gpt-4-0125-preview # gpt-4-turbo-2024-04-09
show_reason=${3:-"True"}  
num_shards=${4:-8} # shards 

total_ex=3000

if [ "$show_reason" = "True" ]; then
    echo "Using checklist" 
    eval_template="evaluation/eval_checklist_score_reason.md"
    eval_folder="evaluation/results_v2_internal/eval_check=${gpt_eval_name}_reason/"
else
    echo "Not showing reason"
    eval_template="evaluation/eval_checklist_score_only.md"
    eval_folder="evaluation/results_v2_internal/eval_check=${gpt_eval_name}/"
fi

mkdir -p $eval_folder

# Decide the shard size dynamically based on $num_shards and the total number 1024
if [ "$num_shards" -eq 1 ]; then
    eval_file="${eval_folder}/${model_name}.json"
    python src/eval.py \
        --action eval \
        --model $gpt_eval_name \
        --max_words_to_eval 1000 \
        --mode checklist \
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
            --mode checklist \
            --eval_template $eval_template \
            --target_model_name $model_name \
            --eval_output_file $eval_file \
            --start_idx $start --end_idx $end  &
    done 
    # Wait for all background processes to finish
    wait

    # # Run the merge results script after all evaluation scripts have completed
    python src/merge_results.py $eval_folder $model_name
fi
 



## V2 checklist eval 
#  bash evaluation/run_checklist_eval.sh gpt-4-turbo-2024-04-09 gpt-4-0125-preview False 10
#  bash evaluation/run_checklist_eval.sh gpt-4-turbo-2024-04-09 gpt-3.5-turbo-0125 False 10
#  bash evaluation/run_checklist_eval.sh gpt-3.5-turbo-0125 gpt-3.5-turbo-0125 False 10