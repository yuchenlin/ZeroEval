model_name=$1 # model to test 
# by default use gpt-3.5-turbo-0125 as ref_name
ref_name=${2:-"gpt-3.5-turbo-0125"} # model to compare 
# by default use "gpt-4-0125-preview" as gpt_eval_name
gpt_eval_name=${3:-"gpt-4-0125-preview"} # evaluator name  # gpt-4-0125-preview
use_checklist=${4:-"True"} 
num_shards=${5:-8} # shards 



if [ "$use_checklist" = "True" ]; then
    echo "Using checklist" 
    eval_template="evaluation/eval_template.md"
    eval_folder="evaluation/results/eval=${gpt_eval_name}/ref=${ref_name}/"
else
    echo "Not using checklist"
    eval_template="evaluation/eval_template.no_checklist.md"
    eval_folder="evaluation/results/eval=${gpt_eval_name}_woCL/ref=${ref_name}/"
fi

mkdir -p $eval_folder

# Decide the shard size dynamically based on $num_shards and the total number 1024
if [ "$num_shards" -eq 1 ]; then
    eval_file="${eval_folder}/${model_name}.json"
    python src/eval.py \
        --action eval \
        --model $gpt_eval_name \
        --max_words_to_eval 1000 \
        --mode pairwise \
        --eval_template $eval_template \
        --target_model_name $model_name \
        --ref_model_name $ref_name \
        --eval_output_file $eval_file 
else
    echo "Using $num_shards shards"
    shard_size=$((1024 / $num_shards))
    echo "Shard size: $shard_size"
    start_gpu=0 # not used 
    for ((start = 0, end = (($shard_size)), gpu = $start_gpu; gpu < $num_shards+$start_gpu; start += $shard_size, end += $shard_size, gpu++)); do
        eval_file="${eval_folder}/${model_name}.$start-$end.json"
        echo "Evaluating $model_name vs $ref_name from $start to $end"
        python src/eval.py \
            --action eval \
            --model $gpt_eval_name \
            --max_words_to_eval 1000 \
            --mode pairwise \
            --eval_template $eval_template \
            --target_model_name $model_name \
            --ref_model_name $ref_name \
            --eval_output_file $eval_file \
            --start_idx $start --end_idx $end \
            &
    done 
    # Wait for all background processes to finish
    wait

    # # Run the merge results script after all evaluation scripts have completed
    python src/merge_results.py $eval_folder $model_name
fi

python src/upload_evaluation.py $gpt_eval_name $ref_name $model_name $use_checklist
# >>>> bash evaluation/run_eval.sh gpt-3.5-turbo-0125 <<<< the reference itself 



# by default, we use "gpt-3.5-turbo-0125" as the reference model
# bash evaluation/run_eval.sh gpt-4-0125-preview
# bash evaluation/run_eval.sh tulu-2-dpo-70b
# bash evaluation/run_eval.sh Mixtral-8x7B-Instruct-v0.1
# bash evaluation/run_eval.sh Mistral-7B-Instruct-v0.2
# bash evaluation/run_eval.sh Yi-34B-Chat
# bash evaluation/run_eval.sh vicuna-13b-v1.5
# bash evaluation/run_eval.sh Llama-2-70b-chat-hf
# bash evaluation/run_eval.sh Llama-2-13b-chat-hf
# bash evaluation/run_eval.sh Llama-2-7b-chat-hf
# bash evaluation/run_eval.sh Mistral-7B-Instruct-v0.1
# bash evaluation/run_eval.sh gemma-7b-it
# bash evaluation/run_eval.sh gemma-2b-it



# Use gpt-4-0125-preview as the reference model
# bash evaluation/run_eval.sh tulu-2-dpo-70b gpt-4-0125-preview
# bash evaluation/run_eval.sh Mixtral-8x7B-Instruct-v0.1 gpt-4-0125-preview
# bash evaluation/run_eval.sh zephyr-7b-beta gpt-4-0125-preview


# Use gpt-4-0125-preview as the reference model and GPT-4 as the judge (using checklist)
# bash evaluation/run_eval.sh claude-3-opus-20240229 gpt-4-0125-preview gpt-4-0125-preview True

# Use gpt-4-0125-preview as the reference model and GPT-4 as the judge (using NO checklist)
# bash evaluation/run_eval.sh claude-3-opus-20240229 gpt-4-0125-preview gpt-4-0125-preview False

# Use gpt-4-0125-preview as the reference model and Claude as the judge (using NO checklist)
# bash evaluation/run_eval.sh claude-3-opus-20240229 gpt-4-0125-preview claude-3-opus-20240229 False 1

# Use gpt-4-0125-preview as the reference model and Claude as the judge (using Checklist)
# bash evaluation/run_eval.sh claude-3-opus-20240229 gpt-4-0125-preview claude-3-opus-20240229 True 1