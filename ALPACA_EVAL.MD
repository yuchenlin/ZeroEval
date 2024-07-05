```bash
bash scripts/_common_eval.sh alpaca_eval Qwen/Qwen2-0.5B-Instruct Qwen2-0.5B-Instruct 4 sampling 0.5 1.0 1.1
bash scripts/_common_eval.sh alpaca_eval yuchenlin/Rex-v0.1-0.5B Rex-v0.1-0.5B 4 sampling 0.5 1.0 1.1

bash scripts/_common_eval.sh alpaca_eval Qwen/Qwen2-1.5B-Instruct Qwen2-1.5B-Instruct 4 sampling 0.5 1.0 1.1
bash scripts/_common_eval.sh alpaca_eval yuchenlin/Rex-v0.1-1.5B Rex-v0.1-1.5B 4 sampling 0.5 1.0 1.1
```


```bash
# model_name="Rex-v0.1-0.5B"
model_name="Qwen2-0.5B-Instruct"
python evaluation/alpaca_eval_formatting.py result_dirs/alpaca_eval/cot=True/${model_name}.json
alpaca_eval --model_outputs result_dirs/alpaca_eval/cot=True/${model_name}.ae.json --reference_outputs result_dirs/alpaca_eval/gpt-3.5-turbo-0301.ae.json
```