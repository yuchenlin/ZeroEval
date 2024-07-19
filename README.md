# ZeroEval: A simple unified framework for evaluating LLMs

ZeroEval is a simple unified framework for evaluating (large) language models on various tasks.
This repository aims to evaluate instruction-tuned LLMs for their zero-shot performance on various reasoning tasks such as MMLU and GSM. We evaluate LLMs with a unified setup by controlling the factors such as prompting, sampling, output parsing, etc. In ZeroEval, we perform **zero-shot** prompting, and instruct LM to output both reasoning and answer in a **json**-formatted output. We are actively adding new tasks. Contributions are welcome! 

- [X post](https://x.com/billyuchenlin/status/1814037110577578377)


## Installation 

<details>
  <summary> Click to expand </summary>

```bash
conda create -n zeroeval python=3.10
conda activate zeroeval
pip install vllm -U # pip install -e vllm 
pip install -r requirements.txt
# export HF_HOME=/path/to/your/custom/cache_dir/ 
```

</details>


## Tasks 

- MMLU-redux (`-d mmlu-redux`)
- GSM (`-d gsm`)
- ZebraLogic (`-d zebra-grid`)
- More tasks will be added soon. (e.g., ARC, MMLU-Pro, etc.)
<!-- - AlpacaEval (`-d alpaca-eval`) -->

## Usage

`zero_eval_local.sh` and `zero_eval_api.sh` are the two main scripts to run the evaluation.

### Examples

- `bash zero_eval_local.sh -d mmlu-redux -m meta-llama/Meta-Llama-3-8B-Instruct -p Meta-Llama-3-8B-Instruct -s 4` (Run Llama-3-8B-Instruct with greedy decoding on `mmlu-redux`)

- `bash zero_eval_api.sh -d gsm -f openai -m openai/gpt-4o-mini-2024-07-18 -p gpt-4o-mini-2024-07-18 -s 8` (Run gpt-4o-mini with greedy decoding on `gsm`)

- `bash zero_eval_api.sh -d zebra-grid -f openai -m deepseek-chat -p deepseek-chat -s 8` (Run deepseek-chat via openai style api, with greedy decoding on `zebra-grid`)


More examples can be found in the `scripts` folder, e.g., the [scripts/_MMLU_redux.md](scripts/_MMLU_redux.md) and [scripts/_GSM.md](scripts/_GSM.md) files.


### Arguments  
 

<details>
<summary>Command Line Arguments</summary>

| Arguments | Description | Default |
|-----|-------------|---------|
| `-d` | DATA_NAME: `mmlu-redux`, `gsm`, `zebra-grid`, `alpaca_eval`, ... (see [src/task_configs.py](src/task_configs.py)) | |
| `-m` | model_name | |
| `-p` | model_pretty_name | |
| `-s` | number of shards (When `-s 1` we'll use all your GPUs for loading the model and running the inference; When `-s K`, we'll use K GPUs and divide the data into K shards for each GPU to run the inference on a single shard, and merge the results at the end.) | 1 |
| `-f` | engine (`vllm` by default for `zero_eval_local.sh`, can be changed to `hf`; For `zero_eval_api.sh`, we can use `openai`, `anthropic`, ...) | `vllm`/`openai` for `zero_eval_local/api.sh` |
| `-r` | run_name (the results will be saved in a sub folder with the `run_name` when it is specified) | "default" |
| `-t` | temperature | 0 (greedy decoding) |
| `-o` | top_p for nucleus sampling | 1.0 |
| `-e` | repetition penalty | 1.0 |
| `-b` | batch size | 4 |

</details>

## Results 


### MMLU-Redux ([full table](result_dirs/mmlu-redux.summary.md))

`python src/evaluation/mcqa_eval.py mmlu-redux`

|             Model              |  Mode  |  Acc  |  No answer  |  Total  |  Reason Lens  |
|--------------------------------|--------|-------|-------------|---------|---------------|
|       gpt-4o-2024-05-13        | greedy | 88.01 |    0.14     |  2778   |    629.79     |
|   claude-3-5-sonnet-20240620   | greedy |  86   |    0.18     |  2778   |    907.09     |
|     gpt-4-turbo-2024-04-09     | greedy | 85.31 |    0.04     |  2778   |    631.38     |
|         gemini-1.5-pro         | greedy | 82.76 |    1.94     |  2778   |     666.7     |
|     claude-3-opus-20240229     | greedy | 82.54 |    0.58     |  2778   |    500.35     |
|        yi-large-preview        | greedy | 82.18 |    0.14     |  2778   |    968.52     |
|           gpt-4-0314           | greedy | 81.64 |    0.04     |  2778   |    397.22     |
|       Qwen2-72B-Instruct       | greedy | 81.57 |    0.25     |  2778   |    486.46     |
|     gpt-4o-mini-2024-07-18     | greedy | 81.5  |    0.07     |  2778   |      526      |
|            yi-large            | greedy | 81.25 |      0      |  2778   |     773.8     |
|         deepseek-chat          | greedy | 80.81 |    0.11     |  2778   |    691.91     |
|         deepseek-coder         | greedy | 79.63 |    0.14     |  2778   |    704.72     |
|   Meta-Llama-3-70B-Instruct    | greedy | 77.97 |    0.11     |  2778   |    520.45     |
|        gemini-1.5-flash        | greedy | 77.36 |    1.26     |  2778   |    583.45     |
|       reka-core-20240501       | greedy | 76.42 |    0.76     |  2778   |     701.4     |
|    gemma-2-27b-it@together     | greedy | 75.67 |    0.61     |  2778   |    446.51     |
|    claude-3-sonnet-20240229    | greedy | 74.87 |    0.07     |  2778   |    671.75     |
|        Yi-1.5-34B-Chat         | greedy | 73.04 |    0.58     |  2778   |    618.29     |
|      gemma-2-9b-it@nvidia      | greedy | 72.82 |    0.76     |  2778   |      499      |
|    claude-3-haiku-20240307     | greedy | 72.32 |    0.04     |  2778   |    644.63     |
|       gpt-3.5-turbo-0125       | greedy | 68.36 |    0.04     |  2778   |    357.92     |
|       Qwen2-7B-Instruct        | greedy | 66.92 |    0.72     |  2778   |    533.15     |
|         Yi-1.5-9B-Chat         | greedy | 65.05 |    4.61     |  2778   |    542.37     |
|      reka-flash-20240226       | greedy | 64.72 |    0.32     |  2778   |    659.25     |
|    Meta-Llama-3-8B-Instruct    | greedy | 61.66 |    0.97     |  2778   |    600.81     |
| Llama-3-Instruct-8B-SimPO-v0.2 | greedy | 55.22 |    1.19     |  2778   |    446.68     |
|      Qwen2-1.5B-Instruct       | greedy | 41.11 |    7.74     |  2778   |    280.53     |


### GSM ([full table](result_dirs/gsm.summary.md))

`python src/evaluation/gsm_eval.py`

|             Model              |  Mode  |  Acc  |  No answer  |  Total  |  Reason Lens  |
|--------------------------------|--------|-------|-------------|---------|---------------|
|   claude-3-5-sonnet-20240620   | greedy | 95.6  |      0      |  1319   |    465.19     |
|     claude-3-opus-20240229     | greedy | 95.6  |      0      |  1319   |    410.62     |
|       gpt-4o-2024-05-13        | greedy | 95.38 |      0      |  1319   |    479.98     |
|     gpt-4o-mini-2024-07-18     | greedy | 94.24 |      0      |  1319   |    463.71     |
|         deepseek-chat          | greedy | 93.93 |      0      |  1319   |    495.52     |
|         deepseek-coder         | greedy | 93.78 |      0      |  1319   |    566.89     |
|         gemini-1.5-pro         | greedy | 93.4  |      0      |  1319   |    389.17     |
|   Meta-Llama-3-70B-Instruct    | greedy | 93.03 |      0      |  1319   |    352.44     |
|       Qwen2-72B-Instruct       | greedy | 92.72 |      0      |  1319   |    375.96     |
|    claude-3-sonnet-20240229    | greedy | 91.51 |      0      |  1319   |    762.69     |
|        gemini-1.5-flash        | greedy | 91.36 |      0      |  1319   |    344.61     |
|    gemma-2-27b-it@together     | greedy | 90.22 |      0      |  1319   |    364.68     |
|    claude-3-haiku-20240307     | greedy | 88.78 |      0      |  1319   |    587.65     |
|       reka-core-20240501       | greedy | 87.49 |    0.08     |  1319   |    414.33     |
|         gemma-2-9b-it          | greedy | 87.41 |      0      |  1319   |    394.83     |
|        Yi-1.5-34B-Chat         | greedy | 84.38 |    0.08     |  1319   |    553.32     |
|            yi-large            | greedy | 81.73 |      0      |  1319   |    479.23     |
|       gpt-3.5-turbo-0125       | greedy | 80.36 |      0      |  1319   |    350.97     |
|       Qwen2-7B-Instruct        | greedy | 80.06 |      0      |  1319   |     452.6     |
|    Meta-Llama-3-8B-Instruct    | greedy | 78.47 |      0      |  1319   |    429.39     |
|         Yi-1.5-9B-Chat         | greedy | 77.86 |    0.08     |  1319   |    485.07     |
|      reka-flash-20240226       | greedy | 74.68 |    0.45     |  1319   |    460.06     |
| Llama-3-Instruct-8B-SimPO-v0.2 | greedy | 57.47 |    2.05     |  1319   |    485.99     |
|      Qwen2-1.5B-Instruct       | greedy | 43.9  |    4.78     |  1319   |    298.07     | 


### ZebraLogic ([full table](result_dirs/zebra-grid.summary.md))

`python src/evaluation/zebra_grid_eval.py`

See the ZebraLogic Leaderboard: [https://huggingface.co/spaces/allenai/ZebraLogic](https://huggingface.co/spaces/allenai/ZebraLogic) 