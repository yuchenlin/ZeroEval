```bash 

# export VLLM_WORKER_MULTIPROC_METHOD="FLASH_ATTN"
# VLLM_WORKER_MULTIPROC_METHOD=spawn 
bash scripts/_common_eval.sh -d zebra-grid -m Qwen/Qwen2-72B-Instruct -p Qwen2-72B-Instruct -s 1
bash scripts/_common_eval.sh -d zebra-grid -m meta-llama/Meta-Llama-3-70B-Instruct -p Meta-Llama-3-70B-Instruct -s 1 
bash scripts/_common_eval.sh -d zebra-grid -m meta-llama/Meta-Llama-3-8B-Instruct -p Meta-Llama-3-8B-Instruct -s 4
bash scripts/_common_eval.sh -d zebra-grid -m Qwen/Qwen2-7B-Instruct -p Qwen2-7B-Instruct -s 4



# pip install flashinfer -i https://flashinfer.ai/whl/cu118/torch2.3/
# VLLM_WORKER_MULTIPROC_METHOD=spawn 
export VLLM_ATTENTION_BACKEND=FLASHINFER; bash scripts/_common_eval.sh -d zebra-grid -m google/gemma-2-9b-it -p gemma-2-9b-it -s 4  -b 1
# export VLLM_ATTENTION_BACKEND=FLASHINFER; bash scripts/_common_eval.sh -d zebra-grid -m google/gemma-2-27b-it -p gemma-2-27b-it -s 1 -b 1 -r "sampling" -t 0.5


bash scripts/_common_api_eval.sh -d zebra-grid -f openai -m google/gemma-2-9b-it@nvidia -p gemma-2-9b-it@nvidia -s 8
bash scripts/_common_api_eval.sh -d zebra-grid -f openai -m google/gemma-2-27b-it@nvidia -p gemma-2-27b-it@nvidia -s 8

# wait 

bash scripts/_common_api_eval.sh -d zebra-grid -f openai -m openai/gpt-3.5-turbo-0125 -p gpt-3.5-turbo-0125 -s 8
bash scripts/_common_api_eval.sh -d zebra-grid -f openai -m openai/gpt-4o-2024-05-13 -p gpt-4o-2024-05-13 -s 8 
bash scripts/_common_api_eval.sh -d zebra-grid -f openai -m openai/gpt-4-turbo-2024-04-09 -p gpt-4-turbo-2024-04-09 -s 8 

bash scripts/_common_api_eval.sh -d zebra-grid -f anthropic -m anthropic/claude-3-5-sonnet-20240620 -p claude-3-5-sonnet-20240620 -s 8
wait
bash scripts/_common_api_eval.sh -d zebra-grid -f anthropic -m anthropic/claude-3-haiku-20240307 -p claude-3-haiku-20240307 -s 8
wait
bash scripts/_common_api_eval.sh -d zebra-grid -f anthropic -m anthropic/claude-3-opus-20240229 -p claude-3-opus-20240229 -s 8
wait
bash scripts/_common_api_eval.sh -d zebra-grid -f anthropic -m anthropic/claude-3-sonnet-20240229 -p claude-3-sonnet-20240229 -s 8
wait

bash scripts/_common_api_eval.sh -d zebra-grid -f google -m google/gemini-1.5-pro -p gemini-1.5-pro -s 8
bash scripts/_common_api_eval.sh -d zebra-grid -f google -m google/gemini-1.5-flash -p gemini-1.5-flash -s 8

bash scripts/_common_api_eval.sh -d zebra-grid -f reka -m reka/reka-flash-20240226 -p reka-flash-20240226 -s 8
bash scripts/_common_api_eval.sh -d zebra-grid -f reka -m reka/reka-core-20240501 -p reka-core-20240501 -s 8

bash scripts/_common_api_eval.sh -d zebra-grid -f together -m meta-llama/Llama-3-70b-chat-hf -p Llama-3-70b-chat-hf -s 8
bash scripts/_common_api_eval.sh -d zebra-grid -f together -m meta-llama/Llama-3-8B-chat-hf -p Llama-3-8B-chat-hf -s 8

bash scripts/_common_api_eval.sh -d zebra-grid -f openai -m deepseek-chat -p deepseek-chat -s 8
bash scripts/_common_api_eval.sh -d zebra-grid -f openai -m deepseek-coder -p deepseek-coder -s 8

bash scripts/_common_api_eval.sh -d zebra-grid -f openai -m yi-large -p yi-large -s 8
bash scripts/_common_api_eval.sh -d zebra-grid -f openai -m yi-large-preview -p yi-large-preview -s 8

```



<!-- 
## ZebraBench (Grid Mode)

```bash 
bash scripts/_common_eval.sh zebra-grid Qwen/Qwen2-72B-Instruct Qwen2-72B-Instruct 1
bash scripts/_common_eval.sh zebra-grid meta-llama/Meta-Llama-3-8B-Instruct Meta-Llama-3-8B-Instruct 4
bash scripts/_common_eval.sh zebra-grid meta-llama/Meta-Llama-3-70B-Instruct Meta-Llama-3-70B-Instruct 1
bash scripts/_common_eval.sh zebra-grid Qwen/Qwen2-7B-Instruct Qwen2-7B-Instruct 4 


bash scripts/_common_eval.sh -d zebra-grid -m Qwen/Qwen2-7B-Instruct -p Qwen2-7B-Instruct -s 4 

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

  -->

