# ZeroEval 


## Motivation

This repository aims to evaluate instruction-tuned LLMs (i.e., chat models instead of base models) for their zero-shot performance on various reasoning tasks such as MMLU. We encourage the model to generate the answer in the form of a natural language sentence, instead of looking at their logits to decide the answer. 

## Tasks 

- MMLU-redux 
- GSM
- AlpacaEval 
- ...


## Arguments for `_common_eval.sh`

- `-d` for DATA_NAME
- `-m` for model_name
- `-p` for model_pretty_name
- `-s` for n_shards
- `-r` for run_name (optional, default is “default”)
- `-t` for TEMP (optional, default is 0)
- `-o` for TOP_P (optional, default is 1.0)
- `-e` for rp (optional, default is 1.0)


## Run  

TBD
