import os
import json  

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

run_name_folders = {
        "greedy": "result_dirs/zebra-grid",
        "sampling": "result_dirs/zebra-grid/sampling",
    } 

model_results = load_model_results(run_name_folders)

for model, filepath in model_results.items():
    print(f"Processing {filepath}")
    with open(filepath, "r") as f:
        data = json.load(f)
        # print(data)
    for item in data:
        del item["solution"]
    
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)