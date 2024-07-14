bash scripts/_common_eval.sh -d mmlu-redux -m 01-ai/Yi-1.5-34B-Chat -p Yi-1.5-34B-Chat -s 1
wait 
bash scripts/_common_eval.sh -d zebra-grid -m 01-ai/Yi-1.5-34B-Chat -p Yi-1.5-34B-Chat -s 1
wait 
bash scripts/_common_eval.sh -d gsm -m 01-ai/Yi-1.5-34B-Chat -p Yi-1.5-34B-Chat -s 1
wait 

bash scripts/_common_eval.sh -d mmlu-redux -m 01-ai/Yi-1.5-9B-Chat -p Yi-1.5-9B-Chat -s 4
wait 
bash scripts/_common_eval.sh -d zebra-grid -m 01-ai/Yi-1.5-9B-Chat -p Yi-1.5-9B-Chat -s 4
wait 
bash scripts/_common_eval.sh -d gsm -m 01-ai/Yi-1.5-9B-Chat -p Yi-1.5-9B-Chat -s 4
wait 