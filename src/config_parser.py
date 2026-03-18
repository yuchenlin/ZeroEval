import argparse

from dataclasses import dataclass
from typing import List

@dataclass
class RunConfig:    
    engine: str 
    model_name: str
    model_pretty_name: str
    tokenizer_name: str

    output_folder: str 
    download_dir: str | None 
    
    dtype: str 
    tokenizer_mode: str 
    data_name: str 
    batch_size: int 
    num_outputs: int
    
    top_p: float 
    temperature: float 
    repetition_penalty: float 
    max_tokens: int 
    max_model_len: int 

    tensor_parallel_size: int 
    data_parallel_size: int
    num_shards: int 
    shard_id: int 
    start_index: int
    end_index: int 
    filepath: str
    cache_filepath: str | None 
    follow_up_mode: str 
    follow_up_file: str | None 
    overwrite: bool
    no_repeat_ngram_size: int
    hf_bf16: bool
    hf_gptq: bool
    gpu_memory_utilization: float 
    use_hf_conv_template: bool 
    use_imend_stop: bool 
    run_name: str 


def sanitize_args(args: RunConfig):
    if args.download_dir == "default":
        args.download_dir = None

    if args.model_pretty_name:
        err = ValueError(f"incorrect value for model_pretty_name: {args.model_pretty_name}")
        # folder name cannot contain any "/" or "\"
        assert args.model_pretty_name.find("/")<0, err
        assert args.model_pretty_name.find("\\")<0, err

    if args.output_folder:
        args.output_folder = args.output_folder.rstrip("/")

    return args

def parse_args(cmd: List[str] | None = None) -> RunConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', default="vllm", type=str)
    parser.add_argument('--output_folder', default="./result_dirs/wild_bench/", type=str)
    parser.add_argument('--download_dir', default=None, type=str)
    parser.add_argument('--model_name', default=None, type=str)
    parser.add_argument('--model_pretty_name', default=None, type=str)
    parser.add_argument('--tokenizer_name', default="auto", type=str)
    parser.add_argument('--tensor_parallel_size', type=int, default=1)
    parser.add_argument('--data_parallel_size', type=int, default=1)
    parser.add_argument('--dtype', type=str, default="bfloat16")
    parser.add_argument('--tokenizer_mode', type=str, default="auto")
    parser.add_argument('--data_name', default="wild_bench", type=str)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--num_outputs', default=1, type=int)
    parser.add_argument('--top_p',default=1, type=float)
    parser.add_argument('--temperature',default=0, type=float)
    parser.add_argument('--repetition_penalty',default=1, type=float)
    parser.add_argument('--max_tokens',default=7500, type=int)
    parser.add_argument('--max_model_len',default=-1, type=int)
    parser.add_argument('--num_shards', default=1, type=int)
    parser.add_argument('--shard_id', default=0, type=int)
    parser.add_argument('--start_index',default=0, type=int) # 0 means from the beginning of the list
    parser.add_argument('--end_index',default=-1, type=int) # -1 means to the end of the list
    parser.add_argument('--filepath',default="auto", type=str)

    parser.add_argument('--cache_filepath', default=None, type=str)

    parser.add_argument('--follow_up_mode', default="N/A", type=str) # N/A means not a follow up
    parser.add_argument('--follow_up_file', default=None, type=str) # if you have an existing file

    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--no_repeat_ngram_size', default=0, type=int)
    parser.add_argument('--hf_bf16', action='store_true')
    parser.add_argument('--hf_gptq', action='store_true')
    parser.add_argument('--gpu_memory_utilization', default=0.9, type=float)

    parser.add_argument('--use_hf_conv_template', action='store_true')
    parser.add_argument('--use_imend_stop', action='store_true')

    # only for MT-bench; not useful for other benchmarks
    # parser.add_argument('--cot', type=str, default="True")
    parser.add_argument('--run_name', type=str, default="")
    args = parser.parse_args(cmd)

    parsed_args = RunConfig(**vars(args))    
    parsed_args = sanitize_args(parsed_args)
    return parsed_args
