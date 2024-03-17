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
    anthropic_chat_request,
)
from datasets import load_dataset, get_dataset_config_names
import tiktoken
 
encoding = None 

def get_args():
    parser = argparse.ArgumentParser()  
    parser.add_argument("--action", type=str, default="trial", required=True)
    parser.add_argument("--mode", type=str, default="pairwise", required=True)
    parser.add_argument("--eval_template", type=str, default="", required=True)
    parser.add_argument("--target_model_name", type=str, required=False) 
    parser.add_argument("--data_name", type=str, default=None)
    parser.add_argument("--ref_model_name", type=str, required=False)
    parser.add_argument("--eval_output_file", type=str, required=True)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=-1)  
    parser.add_argument("--save_interval", type=int, default=3)
    
    # Prompt configs 
    parser.add_argument("--max_words_to_eval", type=int, default=-1)
    
    # OpenAI Configs
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--model", type=str, default="gpt-4-1106-preview")
    parser.add_argument("--engine", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--overwrite", action="store_true")

    
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args() 
    if args.api_key is not None:
        openai.api_key = args.api_key 
     
    return args
        

def parse_result(result_str, mode="json"): 
    result_str = result_str.strip() 
    try: 
        # result_str = result_str.replace(".\n", ". ")
        # result_str = result_str.replace(".\n\n", ". ")
        result_str = result_str.replace("\n", " ")
        parsed_result = json.loads(result_str)
    except Exception as e:
        print(result_str)
        print(e)
        # raise Exception(f"Failed to parse the result: {result_str}")
        parsed_result = {"N/A": "N/A"}
        # exit()
    return parsed_result

def compute_cost(gpt_model_name, prompt, result):
    global encoding
    if encoding is None:
        encoding = tiktoken.encoding_for_model(gpt_model_name)
    if gpt_model_name in ["gpt-4-1106-preview", "gpt-4-0125-preview", "gpt-4-turbo-preview"]:
        price_per_input_token = 0.01 / 1000
        price_per_output_token = 0.03 / 1000
    elif gpt_model_name in ["gpt-3.5-turbo-0125"]:
        price_per_input_token = 0.0005 / 1000
        price_per_output_token = 0.0015 / 1000
    elif gpt_model_name in ["gpt-4"]:
        price_per_input_token = 0.03 / 1000
        price_per_output_token = 0.06 / 1000
    else:
        raise Exception(f"Unknown model: {gpt_model_name}")
    
    price_item = {
                # compute openai token number
                "in_tokens": len(encoding.encode(prompt)),
                "out_tokens": len(encoding.encode(result)),
    }
    price_item["cost"] = price_item["in_tokens"] * price_per_input_token + price_item["out_tokens"] * price_per_output_token
    return price_item


def gpt_eval(results, args): 
    # try to load the existing results from args.eval_output_file 
    if os.path.exists(args.eval_output_file) and not args.overwrite:
        cnt = 0 
        with open(args.eval_output_file, "r") as f:
            existing_results = json.load(f) 
        for i in range(len(existing_results)):
            e = existing_results[i]
            t = results[i]
            if e["prompt"] != t["prompt"]:
                continue
            # if e["prompt"] == t["prompt"] and e["result"] != "N/A":
            #     results[i]["result"] = e["result"]
            #     cnt += 1 
            if "result" in e:
                t["result"] = e["result"]
                if "parsed_result" in e: 
                    t["parsed_result"] = e["parsed_result"]
            if "error" in e:
                t["error"] = e["error"]
            
            if "winner" in e:
                t["winner"] = e["winner"]
                cnt += 1
            
        print(f"loading {cnt} results from {args.eval_output_file}")
     
    # TODO: add support for using other APIs as judge 
    openai_args = {
        "prompt": "TODO",
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "stop": []
    }
    if args.model:
        openai_args['model'] = args.model
    if args.engine:
        openai_args['engine'] = args.engine

    @retry_handler(retry_limit=10)
    def api(ind, item, **kwargs):
        model = kwargs['model']
        if model.startswith('claude'):
            result = anthropic_chat_request(**kwargs)
        else:
            result = openai_chat_request(**kwargs)
        result = result[0]  
        return result
    
    # results = results[args.start_idx:args.end_idx] # for debug
    #import pdb; pdb.set_trace()
    for ind, item in tqdm(enumerate(results), total=len(results), desc=f"Evaluating: {args.eval_output_file} "):
        computed = False
        if item["result"] != "N/A" and item.get("error", "N/A") == "N/A" and "winner" in item:  
            results[ind]["parsed_result"] = parse_result(results[ind]["result"]) 
            computed = True  
            
        openai_args["prompt"] = item["prompt"]
        # if True:
        try:
            if not computed:
                result = api(ind, item, **openai_args)
                results[ind]["result"] = result
            else:
                result = results[ind]["result"]

            results[ind]["parsed_result"] = parse_result(results[ind]["result"])
            r = results[ind]["parsed_result"]

            if args.mode == "pairwise":
                if r["choice"] in ["A", "B"]:
                    results[ind]["winner"] = item["assignment"][r["choice"]]
                elif r["choice"] == "tie":
                    results[ind]["winner"] = "tie"
                else:
                    results[ind]["winner"] = r["choice"] 
            if not args.model.startswith('claude'):
                results[ind]["price"] = compute_cost(args.model, item["prompt"], results[ind]["result"])
            else:
                results[ind]["price"] = {"cost": 0, "in_tokens": 0, "out_tokens": 0}
            results[ind]["error"] = "N/A"
        except Exception as e:
            print(e)
            results[ind]["error"] = str(e)
            results[ind]["result"] = result
            results[ind]["parsed_result"] = {"choice": "N/A"}
            results[ind]["price"] = {"cost": 0, "in_tokens": 0, "out_tokens": 0}
            pass 
        
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
 
    
def placeholder_generation(args, candidates, references, histories, last_queries, checklists): 
    
    with open(args.eval_template) as f:
        eval_template = f.read() 
        print(f"Loaded the eval_template from {args.eval_template}")
    
    results = []
    

    assert len(candidates) == len(references)
            
    L = len(candidates)
    if args.end_idx < 0 or args.end_idx > L:
        args.end_idx = L
 
    print(f"# examples in candidates: {len(candidates)}; We take {args.end_idx-args.start_idx} for evaluation.")
    candidates = candidates[args.start_idx:args.end_idx]
    references = references[args.start_idx:args.end_idx]
    histories = histories[args.start_idx:args.end_idx]
    last_queries = last_queries[args.start_idx:args.end_idx]
    checklists = checklists[args.start_idx:args.end_idx] 
    
    results = []
    for item, ref_item, history, last_query, checklist in zip(candidates, references, histories, last_queries, checklists):
        # print(item, ref_item, history, last_query, checklist)
        o = item["output"][0] if type(item["output"]) == list else item["output"]
        r = ref_item["output"][0] if type(ref_item["output"]) == list else ref_item["output"]
        # random decide which is A and which is B 
        d = {}
        d["session_id"] = item["session_id"]
        d["history"] = history
        d["last_query"] = last_query
        d["model_output"] = item["output"]
        # d["generator"] = args.target_model_name
        d["generator"] = item["generator"]
        if args.mode == "pairwise":
            d["ref_output"] =  r 
            # d["ref_generator"] = args.ref_model_name 
            d["ref_generator"] = ref_item["generator"]
        d["eval_config"] = {"mode": args.mode, "gpt": args.model, "max_words": args.max_words_to_eval}
        
        ## Prompt composition for pairwise evaluation
        if args.mode == "pairwise": 
            if random.random() < 0.5:
                A = o
                B = r
                d["assignment"] = {"A": d["generator"], "B": d["ref_generator"]}
            else:
                A = r
                B = o
                d["assignment"] = {"A": d["ref_generator"], "B": d["generator"]} 
            prompt = eval_template
            prompt = prompt.replace("{$history}", shorten(history, args.max_words_to_eval))
            prompt = prompt.replace("{$user_query}", shorten(last_query, args.max_words_to_eval))
            prompt = prompt.replace("{$candidate_A}", shorten(A, args.max_words_to_eval))
            prompt = prompt.replace("{$candidate_B}", shorten(B, args.max_words_to_eval))
            prompt = prompt.replace("{$checklist}", json.dumps(checklist, indent=2), args.max_words_to_eval)
        
        d["prompt"] = prompt
        if A.strip() == "" and B.strip() == "":
            d["result"] = json.dumps({"reason": "Both responses are empty.", "choice": "tie"})
        elif A.strip() == "":
            d["result"] = json.dumps({"reason": "The response A is empty.", "choice": "B"})
        elif B.strip() == "":
            d["result"] = json.dumps({"reason": "The response B is empty.", "choice": "A"})
        else:
            d["result"] = "N/A" 
        results.append(d)
    return results 

def compose_eval_item(b, t, r, histories, last_queries, checklists):
    assert b["session_id"] == t["session_id"] == r["session_id"]
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

def main():
    args = get_args() 
    random.seed(args.seed)

    # assert the api key is ready 
    if args.model.startswith("claude"):
        assert os.environ.get("ANTHROPIC_API_KEY") is not None, "Please set the ANTHROPIC_API_KEY in the environment variables."
    elif args.model.startswith("gpt"):
        assert os.environ.get("OPENAI_API_KEY") is not None, "Please set the OPENAI_API_KEY in the environment variables."

    if args.action.startswith("trial"):
        results = placeholder_generation(args)
        print(f"We have {len(results)} examples to evaluate!")
        with open(args.eval_output_file, "w") as f:
            json.dump(results, f, indent=2) 
    elif args.action.startswith("eval"):  
        if args.mode != "pairwise":
            raise Exception("Not implemented yet!")
        bench_data = load_dataset("allenai/WildBench", split="test")
        target_model_data = load_dataset("WildEval/WildBench-Results", args.target_model_name, split="train")
        ref_model_data = load_dataset("WildEval/WildBench-Results", args.ref_model_name, split="train")
        histories = []
        last_queries = []
        checklists = []
        for b, t, r in zip(bench_data, target_model_data, ref_model_data):
            compose_eval_item(b, t, r, histories, last_queries, checklists)
        print(f"len(target_model_data)={len(target_model_data)}")
        print(f"len(ref_model_data)={len(ref_model_data)}")
        candidates = list(target_model_data)
        references = list(ref_model_data)    
        results = placeholder_generation(args, candidates, references, histories, last_queries, checklists)
        results = gpt_eval(results, args) 
    elif args.action.startswith("arena"):
        if args.mode != "pairwise":
            raise Exception("Not implemented yet!")
        print("loading the data from WildEval/WildBench")
        bench_data = load_dataset("allenai/WildBench", split="test")
        print("loading the data from WildEval/WildBench-Results")
        model_names = get_dataset_config_names("WildEval/WildBench-Results")
        print(f"model_names={model_names}")
        all_inference_results = {}
        for model_name in tqdm(model_names, desc="Loading the inference results: "):
            all_inference_results[model_name] = list(load_dataset("WildEval/WildBench-Results", model_name, split="train"))
        eval_results = load_dataset("WildEval/WildBench-Evaluation", "all", split="train") 
        covered_eval_ids = [x['eval_id'] for x in eval_results]
        # ["Llama-2-7b-chat-hf.nosp", "Llama-2-13b-chat-hf.nosp", "Llama-2-70b-chat-hf.nosp"] # ["gemini-1.0-pro", "command"]
        must_choose_models = ["mistral-large-2402"] 
        boosting_models = []
        deboosting_models = [] # "gpt-3.5-turbo-0125"
        sampling_weights = {x: 1.0 for x in model_names}
        candidates, references, histories, last_queries, checklists = [], [], [], [], []
        # boosting some models 
        for x in boosting_models:
            sampling_weights[x] *= 2.0
        for x in deboosting_models:
            sampling_weights[x] *= 0.5
        for index, b in tqdm(enumerate(list(bench_data)), desc="Composing the evaluation items: "):
            sid = b["session_id"]
            while True:
                if must_choose_models:
                    sampled_model_1 = random.choices(must_choose_models, k=1)[0]
                else:
                    sampled_model_1 = random.choices(model_names, weights=[sampling_weights[x] for x in model_names], k=1)[0]
                model_names_without_model_1 = [x for x in model_names if x != sampled_model_1]
                sampled_model_2 = random.choices(model_names_without_model_1, k=1)[0]
                eval_id = sid + "-" + sampled_model_1 + "-" + sampled_model_2
                eval_id_ = sid + "-" + sampled_model_2 + "-" + sampled_model_1
                if eval_id not in covered_eval_ids and eval_id_ not in covered_eval_ids:
                    break
            t = all_inference_results[sampled_model_1][index]
            r = all_inference_results[sampled_model_2][index]
            t["generator"] = sampled_model_1
            r["generator"] = sampled_model_2
            candidates.append(t)
            references.append(r)
            compose_eval_item(b, t, r, histories, last_queries, checklists)
            covered_eval_ids.append(eval_id)
        # print(len(candidates), len(references), len(histories), len(last_queries), len(checklists))
        results = placeholder_generation(args, candidates, references, histories, last_queries, checklists)
        # print(f"We have {len(results)} examples to evaluate!")
        results = gpt_eval(results, args)            
                
    else:
        print("Not implemented yet!")

if __name__ == "__main__": 
    main()
    
 
