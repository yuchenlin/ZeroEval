# ZeroEval: A Unified Framework for Evaluating Language Models

ZeroEval is a simple unified framework for evaluating (large) language models on various tasks.
This repository aims to evaluate instruction-tuned LLMs for their zero-shot performance on various reasoning tasks such as MMLU and GSM. We evaluate LLMs with a unified setup by controlling the factors such as prompting, sampling, output parsing, etc. In ZeroEval, we perform **zero-shot** prompting, and instruct LM to output both reasoning and answer in a **json**-formatted output. We are actively adding new tasks. Contributions are welcome! 

- [X post](https://x.com/billyuchenlin/status/1814037110577578377)


## Installation 

<details>
  <summary> Click to expand </summary>

```bash
conda create -n zeroeval python=3.10
conda activate zeroeval
# pip install vllm -U # pip install -e vllm 
pip install vllm==0.5.1
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

- MMLU-Redux: `python src/evaluation/mcqa_eval.py mmlu-redux` --> [Full results](result_dirs/mmlu-redux.summary.md)
- GSM: `python src/evaluation/gsm_eval.py` --> [Full results](result_dirs/gsm.summary.md)
- ZebraLogic: `python src/evaluation/zebra_grid_eval.py` --> [Full results](result_dirs/zebra-grid.summary.md)
  and [Leaderboard](https://huggingface.co/spaces/allenai/ZebraLogic)
- All: `python src/evaluation/summarize.py` --> `result_dirs/summary.md`


| Model                      |   GSM |   MMLU-Redux |   ZebraLogic |   ZebraLogic-Easy |   Average |
|:---------------------------|------:|-------------:|-------------:|------------------:|----------:|
| claude-3-5-sonnet-20240620 | 95.6  |        86    |         33.4 |             87.5  |   75.625  |
| gpt-4o-2024-05-13          | 95.38 |        88.01 |         28.2 |             77.86 |   72.3625 |
| claude-3-opus-20240229     | 95.6  |        82.54 |         27   |             78.21 |   70.8375 |
| deepseek-chat              | 93.93 |        80.81 |         22.7 |             68.57 |   66.5025 |
| Qwen2-72B-Instruct         | 92.72 |        81.57 |         21.4 |             63.93 |   64.905  |
| deepseek-coder             | 93.78 |        79.63 |         21.1 |             64.64 |   64.7875 |
| gpt-4o-mini-2024-07-18     | 94.24 |        81.5  |         20.1 |             62.5  |   64.585  |
| gemini-1.5-pro             | 93.4  |        82.76 |         19.4 |             55.71 |   62.8175 |
| gemini-1.5-flash           | 91.36 |        77.36 |         19.4 |             59.29 |   61.8525 |
| claude-3-sonnet-20240229   | 91.51 |        74.87 |         18.7 |             58.93 |   61.0025 |
| Meta-Llama-3-70B-Instruct  | 93.03 |        77.97 |         16.8 |             52.86 |   60.165  |
| yi-large                   | 81.73 |        81.25 |         18.8 |             58.21 |   59.9975 |
| gemma-2-27b-it             | 90.22 |        75.67 |         16.3 |             50.71 |   58.225  |
| Athene-70B                 | 86.66 |        76.53 |         16.7 |             52.5  |   58.0975 |
| claude-3-haiku-20240307    | 88.78 |        72.32 |         14.3 |             47.86 |   55.815  |
| reka-core-20240501         | 87.49 |        76.42 |         13   |             43.21 |   55.03   |
| gemma-2-9b-it              | 87.41 |        72.82 |         12.8 |             41.79 |   53.705  |
| Yi-1.5-34B-Chat            | 84.38 |        73.04 |         11.5 |             37.5  |   51.605  |
| Mistral-Nemo-Instruct-2407 | 82.79 |        66.88 |         11.8 |             38.93 |   50.1    |
| Meta-Llama-3-8B-Instruct   | 78.47 |        61.66 |         11.9 |             40.71 |   48.185  |
| gpt-3.5-turbo-0125         | 80.36 |        68.36 |         10.1 |             33.57 |   48.0975 |
| Qwen2-7B-Instruct          | 80.06 |        66.92 |          8.4 |             29.29 |   46.1675 |
| reka-flash-20240226        | 74.68 |        64.72 |          9.3 |             30.71 |   44.8525 |
| Yi-1.5-9B-Chat             | 77.86 |        65.05 |          2.3 |              8.21 |   38.355  |

 

## Citation
If you find ZeroEval useful, please cite it as follows in your publication:

```bibtex
@software{Lin_ZeroEval_A_Unified_2024,
    author = {Lin, Bill Yuchen},
    month = jul,
    title = {{ZeroEval: A Unified Framework for Evaluating Language Models}},
    url = {https://github.com/yuchenlin/ZeroEval},
    year = {2024}
}
```