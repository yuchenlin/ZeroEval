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


## Results 

### MMLU-Redux

|             Model              |  Mode  |  Acc  |  No answer  |  Total  |  Reason Lens  |
|--------------------------------|--------|-------|-------------|---------|---------------|
|       gpt-4o-2024-05-13        | greedy | 88.01 |    0.14     |  2778   |    632.69     |
|   claude-3-5-sonnet-20240620   | greedy |  86   |    0.11     |  2778   |    916.86     |
|     gpt-4-turbo-2024-04-09     | greedy | 85.31 |      0      |  2778   |    633.57     |
|         gemini-1.5-pro         | greedy | 82.76 |    1.94     |  2778   |    651.47     |
|     claude-3-opus-20240229     | greedy | 82.54 |      0      |  2778   |    497.86     |
|           gpt-4-0314           | greedy | 81.64 |    0.04     |  2778   |    397.42     |
|       Qwen2-72B-Instruct       | greedy | 81.61 |    0.07     |  2778   |    486.68     |
|            yi-large            | greedy | 81.17 |      0      |  2778   |    775.75     |
|         deepseek-chat          | greedy | 80.81 |    0.11     |  2778   |    692.94     |
|         deepseek-coder         | greedy | 79.63 |    0.11     |  2778   |    705.72     |
|   Meta-Llama-3-70B-Instruct    | greedy | 78.01 |    0.11     |  2778   |    521.34     |
|        gemini-1.5-flash        | greedy | 77.36 |    1.15     |  2778   |    575.73     |
|       reka-core-20240501       | greedy | 76.42 |    0.86     |  2778   |    701.56     |
|    claude-3-sonnet-20240229    | greedy | 74.87 |    0.04     |  2778   |    674.41     |
|      gemma-2-9b-it@nvidia      | greedy | 72.82 |      0      |  2778   |     498.4     |
|    claude-3-haiku-20240307     | greedy | 72.32 |    0.04     |  2778   |    647.72     |
|       gpt-3.5-turbo-0125       | greedy | 68.36 |      0      |  2778   |    357.93     |
|       Qwen2-7B-Instruct        | greedy | 66.92 |    0.58     |  2778   |    533.43     |
|      reka-flash-20240226       | greedy | 64.72 |    0.36     |  2778   |    661.71     |
|    Meta-Llama-3-8B-Instruct    | greedy | 61.66 |    0.97     |  2778   |    601.18     |
| Llama-3-Instruct-8B-SimPO-v0.2 | greedy | 54.82 |    6.01     |  2778   |    433.88     |


### GSM

|             Model              |  Mode  |  Acc  |  No answer  |  Total  |  Reason Lens  |
|--------------------------------|--------|-------|-------------|---------|---------------|
|   claude-3-5-sonnet-20240620   | greedy | 95.6  |      0      |  1319   |    465.19     |
|     claude-3-opus-20240229     | greedy | 95.6  |      0      |  1319   |    410.62     |
|       gpt-4o-2024-05-13        | greedy | 95.38 |      0      |  1319   |    479.98     |
|         deepseek-chat          | greedy | 93.93 |      0      |  1319   |    495.52     |
|         deepseek-coder         | greedy | 93.78 |      0      |  1319   |    566.89     |
|         gemini-1.5-pro         | greedy | 93.4  |      0      |  1319   |    389.17     |
|   Meta-Llama-3-70B-Instruct    | greedy | 93.03 |      0      |  1319   |    352.44     |
|       Qwen2-72B-Instruct       | greedy | 92.72 |      0      |  1319   |    375.96     |
|    claude-3-sonnet-20240229    | greedy | 91.51 |      0      |  1319   |    762.69     |
|        gemini-1.5-flash        | greedy | 91.36 |      0      |  1319   |    344.61     |
|    claude-3-haiku-20240307     | greedy | 88.78 |      0      |  1319   |    587.65     |
|         gemma-2-9b-it          | greedy | 87.41 |      0      |  1319   |    394.83     |
|       gpt-3.5-turbo-0125       | greedy | 80.36 |      0      |  1319   |    350.97     |
|       Qwen2-7B-Instruct        | greedy | 80.06 |    0.08     |  1319   |    452.34     |
|    Meta-Llama-3-8B-Instruct    | greedy | 78.47 |      0      |  1319   |    429.39     |
| Llama-3-Instruct-8B-SimPO-v0.2 | greedy | 50.27 |    21.99    |  1319   |     402.4     |