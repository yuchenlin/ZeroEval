import json 
from collections import defaultdict
import os 
from tabulate import tabulate 
import re
import sys
from eval_utils import load_model_results, extract_values_from_json, extract_last_complete_json
 


def eval_model(model, filepath):
    global private_solutions
    with open(filepath, "r") as f:
        print(f"Processing {filepath}")
        data = json.load(f)

    solved_examples = 0 
    num_total_examples = len(data) 
    no_asnwer = 0  
    
    reason_lens = []
    for item in data:  
        # Read and Parse the prediction from model output
        
        prediction_str = item["output"][0]
        prediction_json = extract_last_complete_json(prediction_str)
        if prediction_json is None or "answer" not in prediction_json:
            prediction_json = extract_values_from_json(prediction_str, allow_no_quotes=True)
        if prediction_json is None or "answer" not in prediction_json or prediction_json["answer"] is None or prediction_json["answer"] == "": 
            no_asnwer += 1 
            if False and  "claude-3-5-sonnet-20240620" in model:
                print(f"No answer for {item['id']}")
                print(prediction_str)
                print(prediction_json)
            continue 
        reason = prediction_json.get("reasoning", "")
        model_answer = prediction_json["answer"]
        correct_answer = item["correct_answer"]
        index_of_correct_answer = item["choices"].index(correct_answer)
        label_of_correct_answer = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[index_of_correct_answer]
        if  model_answer == label_of_correct_answer or f"{label_of_correct_answer})" in model_answer:
            solved_examples += 1
        reason_lens.append(len(reason))
 
    result = {}
    result["Model"] = model.split("%")[0]
    result["Mode"] = model.split("%")[1]
    result["Acc"] = f"{solved_examples/num_total_examples*100:.2f}"
    result["No answer"] = f"{no_asnwer/num_total_examples*100:.2f}"
    result["Total"] = num_total_examples
    result["Reason Lens"] = f"{sum(reason_lens)/len(reason_lens):.2f}"
    return result


def gen_results(run_name_folders): 
    model_results = load_model_results(run_name_folders)

    columns = ["Model", "Mode", "Acc", "No answer", "Total", "Reason Lens"]
    rows = []
    for model_name, filepath in model_results.items(): 
        result = eval_model(model_name, filepath) 
        rows.append(result)

    # sort the rows by puzzle accuracy
    rows = sorted(rows, key=lambda x: -float(x["Acc"]))
    # Convert rows to the expected format for tabulate
    table_data = [[row[col] for col in columns] for row in rows]

    print(tabulate(table_data, headers=columns, tablefmt="fancy_outline", stralign="center", numalign="center"))
    # print(tabulate(rows, headers=columns, tablefmt="github"))

    # write to json file 
    with open(f"result_dirs/{data_name}.summary.json", "w") as f:
        json.dump(rows, f, indent=2)
    
    # write to markdown file
    with open(f"result_dirs/{data_name}.summary.md", "w") as f:
        f.write(tabulate(table_data, headers=columns, tablefmt="github", stralign="center", numalign="center"))

if __name__ == "__main__":
    data_name = sys.argv[1]
    if data_name not in ["mmlu-redux"]:
        print(f"Invalid data name: {data_name}")
        sys.exit(1)
    run_name_folders = {
        "greedy": f"result_dirs/{data_name}", 
    }  
    gen_results(run_name_folders)
