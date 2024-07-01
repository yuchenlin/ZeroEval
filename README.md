# ZeroEval 


## Motivation

This repository aims to evaluate instruction-tuned LLMs (i.e., chat models instead of base models) for their zero-shot performance on various reasoning tasks such as MMLU. We encourage the model to generate the answer in the form of a natural language sentence, instead of looking at their logits to decide the answer. 

## Tasks 

- MMLU-redux 
- GSM8K



## Run 

```bash 
bash scripts/_common_eval.sh mmlu-redux meta-llama/Meta-Llama-3-8B-Instruct Meta-Llama-3-8B-Instruct 4
# [DATA_NAME]  [model_name]  [model_pretty_name]  [n_shards]
```


## ZebraBench 
```bash 
bash scripts/_common_eval.sh zebra-grid meta-llama/Meta-Llama-3-8B-Instruct Meta-Llama-3-8B-Instruct 4
bash scripts/_common_eval.sh zebra-grid meta-llama/Meta-Llama-3-70B-Instruct Meta-Llama-3-70B-Instruct 1
bash scripts/_common_eval.sh zebra-grid Qwen/Qwen2-72B-Instruct Qwen2-72B-Instruct 1
bash scripts/_common_eval.sh zebra-grid Qwen/Qwen2-7B-Instruct Qwen2-7B-Instruct 4
wait 
bash scripts/_common_eval.sh zebra-grid deepseek-ai/DeepSeek-Coder-V2-Instruct DeepSeek-Coder-V2-Instruct 1
# wait 


###
bash scripts/_common_api_eval.sh zebra-grid openai openai/gpt-3.5-turbo-0125 gpt-3.5-turbo-0125 8
bash scripts/_common_api_eval.sh zebra-grid openai openai/gpt-4o-2024-05-13 gpt-4o-2024-05-13 8 
bash scripts/_common_api_eval.sh zebra-grid openai openai/gpt-4-turbo-2024-04-09 gpt-4-turbo-2024-04-09 8 
bash scripts/_common_api_eval.sh zebra-grid openai openai/gpt-4-0314 gpt-4-0314 8 

bash scripts/_common_api_eval.sh zebra-grid anthropic anthropic/claude-3-haiku-20240307 claude-3-haiku-20240307 8
bash scripts/_common_api_eval.sh zebra-grid anthropic anthropic/claude-3-sonnet-20240229 claude-3-sonnet-20240229 8
bash scripts/_common_api_eval.sh zebra-grid anthropic anthropic/claude-3-opus-20240229 claude-3-opus-20240229 8
bash scripts/_common_api_eval.sh zebra-grid anthropic anthropic/claude-3-5-sonnet-20240620 claude-3-5-sonnet-202401620 8

bash scripts/_common_api_eval.sh zebra-grid google google/gemini-1.5-pro gemini-1.5-pro 8
bash scripts/_common_api_eval.sh zebra-grid google google/gemini-1.5-flash gemini-1.5-flash 8



bash scripts/_common_api_eval.sh zebra-grid reka reka/reka-flash-20240226 reka-flash-20240226 8
bash scripts/_common_api_eval.sh zebra-grid reka reka/reka-core-20240501 reka-core-20240501 8

```

<!-- 


bash scripts/Magpie-Pro-SFT-v0.1.sh mmlu-redux false
wait 
bash scripts/Meta-Llama-3-8B-Instruct.sh mmlu-redux false
wait 
bash scripts/Llama-3-8B-WildChat.sh mmlu-redux false
wait 
bash scripts/Llama-3-8B-Tulu-330K.sh mmlu-redux false
wait 
bash scripts/Llama-3-8B-OpenHermes-243K.sh  mmlu-redux false
wait 
bash scripts/Llama-3-8B-Ultrachat-200K.sh mmlu-redux false
wait 
bash scripts/Llama-3-8B-WizardLM-196K.sh mmlu-redux false

bash scripts/Llama-3-8B-Magpie-Pro-SFT-200K-v0.1.sh mmlu-redux false
bash scripts/Llama-3-8B-Magpie-Pro-SFT-100K-v0.1.sh mmlu-redux false
bash scripts/Llama-3-8B-Magpie-Air-SFT-v0.1.sh mmlu-redux false
 -->
