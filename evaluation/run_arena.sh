run_name=${1:-"round1"}
seed=${2:-"42"}
gpt_eval_name="gpt-4-0125-preview"

eval_folder="evaluation/results/eval=${gpt_eval_name}/arena/"
mkdir -p $eval_folder


n_shards=8
shard_size=128
start_gpu=0
for ((start = 0, end = (($shard_size)), gpu = $start_gpu; gpu < $n_shards+$start_gpu; start += $shard_size, end += $shard_size, gpu++)); do
    eval_file="${eval_folder}/${run_name}.$start-$end.json"
    python src/eval.py \
        --action arena \
        --model $gpt_eval_name \
        --max_words_to_eval 1000 \
        --mode pairwise \
        --seed $seed \
        --eval_template evaluation/eval_template.md \
        --eval_output_file $eval_file \
        --start_idx $start --end_idx $end &
done

# Wait for all background processes to finish
wait

# # Run the merge results script after all evaluation scripts have completed
python src/merge_results.py $eval_folder $run_name
python src/upload_evaluation.py $gpt_eval_name "arena" $run_name
# >>>> bash evaluation/run_eval.sh gpt-3.5-turbo-0125 <<<< the reference itself  

# bash evaluation/run_arena.sh round1 42
# bash evaluation/run_arena.sh round2 1337
