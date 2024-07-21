bash zero_eval_local.sh -d gsm -m microsoft/Phi-3-mini-4k-instruct -p Phi-3-mini-4k-instruct -s 4
wait 
bash zero_eval_local.sh -d mmlu-redux -m microsoft/Phi-3-mini-4k-instruct -p Phi-3-mini-4k-instruct -s 4
wait 
bash zero_eval_local.sh -d zebra-grid -m microsoft/Phi-3-mini-4k-instruct -p Phi-3-mini-4k-instruct -s 4
wait 


# bash zero_eval_local.sh -d mmlu-redux -m 01-ai/Yi-1.5-9B-Chat -p Yi-1.5-9B-Chat -s 4
# wait 
# bash zero_eval_local.sh -d zebra-grid -m 01-ai/Yi-1.5-9B-Chat -p Yi-1.5-9B-Chat -s 4
# wait 
# bash zero_eval_local.sh -d gsm -m 01-ai/Yi-1.5-9B-Chat -p Yi-1.5-9B-Chat -s 4
# wait 