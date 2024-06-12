# ZeroEval 


## Motivation

This repository aims to evaluate instruction-tuned LLMs (i.e., chat models instead of base models) for their zero-shot performance on various reasoning tasks such as MMLU. We encourage the model to generate the answer in the form of a natural language sentence, instead of looking at their logits to decide the answer. 

## Tasks 

- MMLU-redux 
- GSM8K



## Run 

```bash 
bash scripts/gpt-3.5-turbo-0125.sh mmlu-redux false # for using a simpel template
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
 -->
