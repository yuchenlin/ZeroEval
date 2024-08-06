bash zero_eval_api.sh -f openai -d gsm -m openai/gpt-4o-2024-08-06 -p gpt-4o-2024-08-06 -s 8
wait
bash zero_eval_api.sh -f openai -d crux -m openai/gpt-4o-2024-08-06 -p gpt-4o-2024-08-06 -s 8
wait
bash zero_eval_api.sh -f openai -d zebra-grid -m openai/gpt-4o-2024-08-06 -p gpt-4o-2024-08-06 -s 8
wait
bash zero_eval_api.sh -f openai -d mmlu-redux -m openai/gpt-4o-2024-08-06 -p gpt-4o-2024-08-06 -s 8
wait
