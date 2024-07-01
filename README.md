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
