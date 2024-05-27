

# 🦁 WildBench: Benchmarking LLMs with Challenging Tasks from Real Users in the Wild (v2)

<div style="display: flex; justify-content: flex-start;"><img src="https://github.com/allenai/WildBench/blob/main/docs/gray_banner.png?raw=true" alt="Banner" style="width: 40vw; min-width: 300px; max-width: 800px;"> </div>


## Quick Links:
- [HF Leaderboard](https://huggingface.co/spaces/allenai/WildBench)
- [HF Dataset](https://huggingface.co/datasets/allenai/WildBench)


## Installation

<!-- 
conda create -p /net/nfs/mosaic/yuchenl/envs/wbnfs python=3.10 
conda activate /net/nfs/mosaic/yuchenl/envs/wbnfs
-->
```bash
conda create -n wildbench python=3.10
conda activate wildbench
pip install vllm -U # pip install -e vllm 
pip install openai datasets tenacity
pip install google-cloud-aiplatform cohere mistralai 
pip install anthropic==0.19.0
# export HF_HOME=/path/to/your/custom/cache_dir/
```

<!-- 
pip install vllm==0.3.1
pip install openai==0.28.0
pip install datasets tenacity

export HF_HOME=/net/nfs/climate/tmp_cache/
 -->


## How to add a new model to 🦁 WildBench benchmark 

> [!NOTE]
> If your model is on HuggingFace and/or it is supported by [vLLM](https://github.com/vllm-project/vllm), please create an **Issue** here to tell us your model id, chat template, and your preferred sampling parameters. We will add the script to run your model to the repo here and run inference & evaluation for you. If you'd like to try to run inference on your model yourself or you'd like to create a PR for adding your model here, you can follow the instructions below. 

### Case 1: Models supported by vLLM

You can take the files under `scripts` as a reference to add a new model to the benchmark, for example, to add `zephyr-7b-beta` to the benchmark, you can follow the following steps:
1. Create a script named "zephyr-7b-beta.py" under `scripts` folder.
2. Copy and paste the most similar existing script file to it.  For example, `Mistral-7B-Instruct-v0.1.sh` is the most similar to `zephyr-7b-beta.py`.
3. Change the `model_name` and `model_pretty_name` to `HuggingFaceH4/zephyr-7b-beta` and `zephyr-7b-beta` respectively. Make sure that `model_name` is the same as the model name in the Hugging Face model hub, and the `model_pretty_name` is the same as the script name without the `.py` extension.
4. Specify the conversation template for this model by modifying the code in `src/fastchat_conversation.py`.
5. Run your script to make sure it works. You can run the script by running `bash scripts/zephyr-7b-beta.sh` in the root folder.

### Case 2: Models that are only supported by native HuggingFace API

Some new models may not be supported by vLLM for now. You can do the same thing as above but use `--engine hf` in the script instead, and test your script. Note that some models may need more specific configurations, and you will need to read the code and modify them accordingly. In these cases, you should add name-checking conditions to ensure that the model-specific changes are only applied to the specific model.

### Case 3: Private API-based Models

You should change the code to add these APIs, for example, gemini, cohere, claude, and reka. You can refer to the `--engine openai` logic in the existing scripts to add your own API-based models. Please make sure that you do not expose your API keys in the code.



### Next: How to run the benchmark

You can run the benchmark by running the scripts under `scripts` folder. For example, `bash scripts/zephyr-7b-beta.sh`. This will generate either a single result file in the specified `output_dir`. Note that if you use the shard mode, the script will merge the results into a single file later when all subprocesses are finished.


After you tested your script, you can create a PR so we will run your script on our side and merge it into the benchmark. (Note that we may need to modify the script to make it work on our side, and we will let you know if we need to do so. Also, due to the non-deterministic nature of LLMs, our results may not be the same as yours. )









