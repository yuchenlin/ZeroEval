import json 
from collections import defaultdict
import os 
from tabulate import tabulate 
import re
 
def load_model_results(run_name_folders):
    model_results = {}
    for run_name, folder in run_name_folders.items():
        # iterate all json files under the folder 
        for filename in os.listdir(folder):
            filepath = os.path.join(folder, filename)
            if not filename.endswith(".json"):
                continue
            model_name = filename.replace(".json", "")  
            model_name = f"{model_name}%{run_name}"
            model_results[model_name] = filepath  
    return model_results
 

def extract_values_from_json(json_string, keys = ["reasoning", "answer"]):
    extracted_values = {}
    for key in keys:
        # Create a regular expression pattern to find the value for the given key
        pattern = f'"{key}"\\s*:\\s*"([^"]*?)"'
        match = re.search(pattern, json_string)
        if match:
            extracted_values[key] = match.group(1)
        else:
            # Handle the case where the value might contain broken quotes
            pattern = f'"{key}"\\s*:\\s*"(.*?)"'
            match = re.search(pattern, json_string, re.DOTALL)
            if match:
                extracted_values[key] = match.group(1)
    return extracted_values

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
        prediction_json = extract_values_from_json(prediction_str)
        if prediction_json is None or "answer" not in prediction_json: 
            no_asnwer += 1 
            print(f"No answer for {item['id']}")
            # print(prediction_str)
            # print(prediction_json)
            continue 
        reason = prediction_json.get("reasoning", "")
        model_answer = prediction_json["answer"]
        correct_answer = item["correct_answer"]
        index_of_correct_answer = item["choices"].index(correct_answer)
        label_of_correct_answer = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[index_of_correct_answer]
        if model_answer == label_of_correct_answer:
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


if __name__ == "__main__":
    data_name = "mmlu-redux"
    run_name_folders = {
        "greedy": f"result_dirs/{data_name}", 
    }  
    gen_results(run_name_folders)
