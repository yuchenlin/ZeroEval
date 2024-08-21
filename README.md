# ZeroEval: A Unified Framework for Evaluating Language Models

ZeroEval is a simple unified framework for evaluating (large) language models on various tasks.
This repository aims to evaluate instruction-tuned LLMs for their zero-shot performance on various reasoning tasks such as MMLU and GSM. We evaluate LLMs with a unified setup by controlling the factors such as prompting, sampling, output parsing, etc. In ZeroEval, we perform **zero-shot** prompting, and instruct LM to output both reasoning and answer in a **json**-formatted output. We are actively adding new tasks. Contributions are welcome! 

- Leaderboard: [https://hf.co/spaces/allenai/ZeroEval](https://huggingface.co/spaces/allenai/ZeroEval)
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

- [MMLU-redux](https://arxiv.org/abs/2406.04127) (`-d mmlu-redux`)
- [GSM8K](https://openai.com/index/solving-math-word-problems/) (`-d gsm`)
- [ZebraLogic](https://huggingface.co/blog/yuchenlin/zebra-logic) (`-d zebra-grid`)
- [CRUX](https://crux-eval.github.io/) (`-d crux`)
- More tasks will be added soon. (e.g., ARC, MMLU-Pro, etc.)
<!-- - AlpacaEval (`-d alpaca-eval`) -->

## Usage

`zero_eval_local.sh` and `zero_eval_api.sh` are the two main scripts to run the evaluation.

### Examples

- `bash zero_eval_local.sh -d mmlu-redux -m meta-llama/Meta-Llama-3-8B-Instruct -p Meta-Llama-3-8B-Instruct -s 4` (Run Llama-3-8B-Instruct with greedy decoding on `mmlu-redux`)

- `bash zero_eval_api.sh -d gsm -f openai -m openai/gpt-4o-mini-2024-07-18 -p gpt-4o-mini-2024-07-18 -s 8` (Run gpt-4o-mini with greedy decoding on `gsm`)

- `bash zero_eval_api.sh -d zebra-grid -f openai -m deepseek-chat -p deepseek-chat -s 8` (Run deepseek-chat via openai style api, with greedy decoding on `zebra-grid`)


More examples can be found in the `scripts` folder, e.g., the [scripts/_MMLU_redux.md](scripts/_MMLU_redux.md) and [scripts/_GSM.md](scripts/_GSM.md) files as well as [`scripts/local/crux.sh`](ZeroEval/scripts/local/crux.sh).

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
- CRUX: `python src/evaluation/crux_eval.py` --> [Full results](result_dirs/crux.summary.md)
- All: `python src/evaluation/summarize.py` --> [Full results](result_dirs/summary.md) ⬇️

| Model                       |   GSM8K |   MMLU<br/>-Redux |   ZebraLogic<br/>-Easy |   CRUX |   Average |
|:----------------------------|--------:|------------------:|-----------------------:|-------:|----------:|
| claude-3-5-sonnet-20240620  |   95.60 |             86.00 |                  87.50 |  78.75 |     86.96 |
| gpt-4o-2024-05-13           |   95.38 |             88.01 |                  77.86 |  83.62 |     86.22 |
| Llama-3.1-405B-Inst-fp8     |   95.91 |             85.64 |                  87.14 |  72.12 |     85.20 |
| Mistral-Large-2             |   95.53 |             82.97 |                  80.36 |  72.88 |     82.94 |
| gemini-1.5-pro-exp-0801     |   95.00 |             85.53 |                  72.50 |  73.00 |     81.51 |
| claude-3-opus-20240229      |   95.60 |             82.54 |                  78.21 |  68.62 |     81.24 |
| Meta-Llama-3.1-70B-Instruct |   94.16 |             82.97 |                  73.57 |  62.62 |     78.33 |
| deepseek-v2-chat-0628       |   93.93 |             80.81 |                  68.57 |  68.50 |     77.95 |
| gpt-4o-mini-2024-07-18      |   94.24 |             81.50 |                  62.50 |  73.50 |     77.94 |
| deepseek-v2-coder-0724      |   91.51 |             80.24 |                  61.79 |  67.75 |     75.32 |
| Qwen2-72B-Instruct          |   92.65 |             81.61 |                  63.93 |  57.38 |     73.89 |
| claude-3-sonnet-20240229    |   91.51 |             74.87 |                  58.93 |  64.75 |     72.52 |
| gemini-1.5-flash            |   91.36 |             77.36 |                  59.29 |  61.88 |     72.47 |
| yi-large-preview            |   82.64 |             82.15 |                  58.93 |  58.63 |     70.59 |
| Meta-Llama-3-70B-Instruct   |   93.03 |             78.01 |                  52.86 |  57.12 |     70.26 |
| yi-large                    |   80.06 |             81.17 |                  58.21 |  58.38 |     69.46 |
| gemma-2-27b-it              |   90.22 |             75.67 |                  50.71 |  55.88 |     68.12 |
| claude-3-haiku-20240307     |   88.78 |             72.32 |                  47.86 |  53.62 |     65.64 |
| reka-core-20240501          |   87.41 |             76.42 |                  43.21 |  45.00 |     63.01 |
| gemma-2-9b-it               |   87.41 |             72.82 |                  41.79 |  44.88 |     61.72 |
| Yi-1.5-34B-Chat             |   84.08 |             72.79 |                  37.50 |  42.88 |     59.31 |
| gpt-3.5-turbo-0125          |   80.36 |             68.36 |                  33.57 |  53.25 |     58.88 |
| Meta-Llama-3.1-8B-Instruct  |   84.00 |             67.24 |                  43.57 |  38.75 |     58.39 |
| Phi-3-mini-4k-instruct      |   75.51 |             70.34 |                  38.21 |  43.50 |     56.89 |
| Meta-Llama-3-8B-Instruct    |   78.47 |             61.66 |                  40.71 |  36.62 |     54.37 |
| Qwen2-7B-Instruct           |   80.06 |             66.92 |                  29.29 |  36.75 |     53.26 |
| Mixtral-8x7B-Instruct-v0.1  |   70.13 |             63.17 |                  28.93 |  43.50 |     51.43 |
| reka-flash-20240226         |   74.68 |             64.72 |                  30.71 |  33.25 |     50.84 |
| Yi-1.5-9B-Chat              |   76.42 |             65.05 |                   8.21 |  43.75 |     48.36 |
| gemma-2-2b-it               |   51.63 |             51.94 |                  14.29 |  20.75 |     34.65 |

### Changelogs 

- 08/02/2024: added Gemini 1.5 Pro Exp 0801 and CRUX results 
- 07/31/2024: added Meta-Llama-3.1-70B-Instruct and gemma-2-2b-it 
- 07/29/2024: added Llama-3.1-8B, Mistral-Large-2, and deepseek-coder-v2-0724 

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

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=yuchenlin/ZeroEval&type=Date)](https://star-history.com/#yuchenlin/ZeroEval&Date)
