export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ATTENTION_BACKEND=FLASHINFER
bash scripts/_common_eval.sh -d mmlu-redux -m google/gemma-2-27b-it -p gemma-2-27b-it -s 1
bash scripts/_common_eval.sh -d gsm -m google/gemma-2-27b-it -p gemma-2-27b-it -s 1
bash scripts/_common_eval.sh -d zebra-grid -m google/gemma-2-27b-it -p gemma-2-27b-it -s 1