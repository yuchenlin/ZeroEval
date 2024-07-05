from datasets import load_dataset
from _TEMPLATES import apply_mc_template, apply_lgp_grid_template

def mapping_task_names(data_name):
    if data_name == "mmlu-redux":
        dataset = load_dataset("yuchenlin/zero-eval", "mmlu-redux", split="test")
    elif data_name == "gsm":
        dataset = load_dataset("yuchenlin/zero-eval", "gsm", split="test")
    elif data_name == "zebra-grid":
        dataset = load_dataset("allenai/ZebraLogicBench", "grid_mode", split="test")
    else:
        raise ValueError(f"Data name {data_name} not supported")
    return dataset 

def prompt_generation(data_name, data_item, args):
    if data_name in ["mmlu-redux"]:  # and other multiple-choice QA dataset 
        prompt = apply_mc_template(data_item, cot = args.cot) 
    elif data_name in ["zebra-grid"]:
        prompt = apply_lgp_grid_template(data_item, cot = args.cot) 
    else:
        raise ValueError(f"Data name {data_name} not supported")
    return prompt