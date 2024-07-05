
## ZebraBench (Grid Mode)

```bash 
bash scripts/_common_eval.sh zebra-grid Qwen/Qwen2-72B-Instruct Qwen2-72B-Instruct 1
bash scripts/_common_eval.sh zebra-grid meta-llama/Meta-Llama-3-8B-Instruct Meta-Llama-3-8B-Instruct 4
bash scripts/_common_eval.sh zebra-grid meta-llama/Meta-Llama-3-70B-Instruct Meta-Llama-3-70B-Instruct 1
bash scripts/_common_eval.sh zebra-grid Qwen/Qwen2-7B-Instruct Qwen2-7B-Instruct 4 

# wait 


###
bash scripts/_common_api_eval.sh zebra-grid openai openai/gpt-3.5-turbo-0125 gpt-3.5-turbo-0125 8
bash scripts/_common_api_eval.sh zebra-grid openai openai/gpt-4o-2024-05-13 gpt-4o-2024-05-13 8 
bash scripts/_common_api_eval.sh zebra-grid openai openai/gpt-4-turbo-2024-04-09 gpt-4-turbo-2024-04-09 8 

bash scripts/_common_api_eval.sh zebra-grid anthropic anthropic/claude-3-5-sonnet-20240620 claude-3-5-sonnet-20240620 8
wait
bash scripts/_common_api_eval.sh zebra-grid anthropic anthropic/claude-3-haiku-20240307 claude-3-haiku-20240307 8
wait
bash scripts/_common_api_eval.sh zebra-grid anthropic anthropic/claude-3-opus-20240229 claude-3-opus-20240229 8
wait
bash scripts/_common_api_eval.sh zebra-grid anthropic anthropic/claude-3-sonnet-20240229 claude-3-sonnet-20240229 8
wait

bash scripts/_common_api_eval.sh zebra-grid google google/gemini-1.5-pro gemini-1.5-pro 8
bash scripts/_common_api_eval.sh zebra-grid google google/gemini-1.5-flash gemini-1.5-flash 8


bash scripts/_common_api_eval.sh zebra-grid reka reka/reka-flash-20240226 reka-flash-20240226 8
bash scripts/_common_api_eval.sh zebra-grid reka reka/reka-core-20240501 reka-core-20240501 8


bash scripts/_common_api_eval.sh zebra-grid together meta-llama/Llama-3-70b-chat-hf Llama-3-70b-chat-hf 8
bash scripts/_common_api_eval.sh zebra-grid together meta-llama/Llama-3-8B-chat-hf Llama-3-8B-chat-hf 8


bash scripts/_common_api_eval.sh zebra-grid openai deepseek-chat deepseek-chat 8
bash scripts/_common_api_eval.sh zebra-grid openai deepseek-coder deepseek-coder 8

bash scripts/_common_api_eval.sh zebra-grid openai yi-large yi-large 8
bash scripts/_common_api_eval.sh zebra-grid openai yi-large-preview yi-large-preview 8
```

 