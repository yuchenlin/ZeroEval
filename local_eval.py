"""
This script is used for local evaluation of the generated responses by assessing a model output on each question in the checklist, and average its scores.
The code is modified from the `eval.py` which focuses on pairwise evaluaiton.
"""
import argparse
import os
import json
import openai 
import random
from pathlib import Path
from itertools import combinations
from string import Template
from tqdm import tqdm
from threading import get_ident
from concurrent.futures import ThreadPoolExecutor
from eval_utils import (
    retry_handler, 
    openai_chat_request, 
)
from datasets import load_dataset
import tiktoken
 
encoding = None 

def get_args():
    parser = argparse.ArgumentParser() 
    
    parser.add_argument("--action", type=str, default="trial", required=True)
    parser.add_argument("--mode", type=str, default="single", required=True)
    parser.add_argument("--eval_template", type=str, default="", required=True)
    parser.add_argument("--target_model_name", type=str, required=True) 
    parser.add_argument("--data_name", type=str, default=None) 
    parser.add_argument("--eval_output_file", type=str, required=True)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=-1)  
    parser.add_argument("--save_interval", type=int, default=3)
    
    # Prompt configs 
    parser.add_argument("--max_words_to_eval", type=int, default=-1)
    
    # Evaluator Configs 
    parser.add_argument("--evaluator_model", type=str, default="") # we can use gemma-2(-it) for trial run and developing and then move to other better LLMs 
    parser.add_argument("--engine", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--overwrite", action="store_true")
    
    args = parser.parse_args()  
    return args
        

def parse_result(result_str, mode="json"): 
    result_str = result_str.strip() 
    try: 
        parsed_result = json.loads(result_str)
    except Exception as e:
        # print(e)
        # raise Exception(f"Failed to parse the result: {result_str}")
        parsed_result = {"N/A": "N/A"}
        # exit()
    return parsed_result
 

def local_eval(results, args):   
    # results = results[args.start_idx:args.end_idx] # for debug
    for ind, item in tqdm(enumerate(results), total=len(results), desc=f"Evaluating: {args.eval_output_file} "):
        computed = False
        if item["result"] != "N/A" or item.get("error", "N/A") != "N/A": 
            results[ind]["parsed_result"] = parse_result(results[ind]["result"]) 
            computed = True  
            
        # TODO: the Local Evaluation 
        
        # print("Done!") 
        if ind % args.save_interval == 0 or ind == len(results)-1:
            with open(args.eval_output_file, "w") as f:
                json.dump(results, f, indent=2) 
    with open(args.eval_output_file, "w") as f:
        json.dump(results, f, indent=2)
    return results 

def shorten(text, K=-1):
    if K > 0 and len(text.split(" ")) > K:
        text = " ".join(text.split(" ")[:K]) + "... (truncated)" 
    return text
 
    
def placeholder_generation(args): 
    
    with open(args.eval_template) as f:
        eval_template = f.read() 
        print(f"Loaded the eval_template from {args.eval_template}")
    
    results = [] 
         
    bench_data = load_dataset("WildEval/WildBench", split="test")
    target_model_data = load_dataset("WildEval/WildBench-Results", args.target_model_name, split="train") 
    histories = []
    last_queries = []
    checklists = []
    for b, t in zip(bench_data, target_model_data):
        assert b["session_id"] == t["session_id"] == f["session_id"]
        history = ""
        checklist = b["checklist"]
        if len(b["conversation_input"]) > 0: 
            for x in b["conversation_input"][:-1]:
                if x["role"] == "user":
                    history += "USER: " + x["content"] + "\n\n"
                elif x["role"] == "assistant":
                    history += "ASSISTANT: " + x["content"] + "\n\n"
        last_query = b["conversation_input"][-1]["content"]
        histories.append(history)
        last_queries.append(last_query)
        checklists.append(checklist)  
    print(f"len(target_model_data)={len(target_model_data)}") 

    candidates = list(target_model_data)  

    L = len(candidates)
    if args.end_idx < 0 or args.end_idx > L:
        args.end_idx = L
 
    print(f"# examples in candidates: {len(candidates)}; We take {args.end_idx-args.start_idx} for evaluation.")
    candidates = candidates[args.start_idx:args.end_idx] 
    histories = histories[args.start_idx:args.end_idx]
    last_queries = last_queries[args.start_idx:args.end_idx]
    checklists = checklists[args.start_idx:args.end_idx] 
    
    results = []
    for item, history, last_query, checklist in zip(candidates, histories, last_queries, checklists):
        # print(item, ref_item, history, last_query, checklist)
        o = item["output"][0] if type(item["output"]) == list else item["output"] 
        # random decide which is A and which is B 
        d = {}
        d["session_id"] = item["session_id"]
        d["history"] = history
        d["last_query"] = last_query
        d["model_output"] = item["output"]
        d["generator"] = args.target_model_name 
        d["eval_config"] = {"mode": args.mode, "gpt": args.model, "max_words": args.max_words_to_eval}
        
        if args.mode == "single":
            # TODO: the single scoring mode
            pass 
            prompt = "TODO" # based on `eval_template` and `item` (especially the checklist) 
            # we should look at each checklist question and generate a prompt for it. 
        d["prompt"] = prompt
        results.append(d)
    return results 


def main():
    random.seed(42)
    args = get_args() 
    if args.action.startswith("trial"):
        results = placeholder_generation(args)
        print(f"We have {len(results)} examples to evaluate!")
        with open(args.eval_output_file, "w") as f:
            json.dump(results, f, indent=2) 
    elif args.action.startswith("eval"):
        results = placeholder_generation(args)
        results = local_eval(results, args) 
    else:
        print("Not implemented yet!")

if __name__ == "__main__": 
    main()
    
 