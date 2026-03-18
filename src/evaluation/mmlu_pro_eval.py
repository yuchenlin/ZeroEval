import json 
from collections import defaultdict
import os 
from tabulate import tabulate 
import re
import sys 
from src.evaluation.eval_utils import load_model_results, model_name_replacement
from src.tasks import MMLUProTask, MMLUProShortTask
from typing import List

DEBUG = False

TASK_MAP = {
    MMLUProTask.name: MMLUProTask(),
    MMLUProShortTask.name: MMLUProShortTask(),
}


def eval_model(model, filepath):

    with open(filepath, "r") as f:        
        data = json.load(f)

    solved_examples = 0 
    num_total_examples = len(data) 
    no_answer = 0  

    # solved in categories
    category_correct = defaultdict(int)
    category_total = defaultdict(int)
    
    reason_lens = []
    parsed_results = [] 
    for item in data:  
        
        # Read and Parse the prediction from model output
        parsed_item = item.copy()
        output = item["output"][0]
        category = item["category"]
        correct_answer: str = item["answer"]
        reason = output # as proxy full output with reasoning        

        category_total[category] += 1

        predictions = re.findall(r"```json.*?```|\{.*?\}", output, flags=re.DOTALL)

        model_answer = None
        if predictions:
            text = predictions[-1]
            text = re.sub(r'\\\\', r'\\', text)
            match = re.search("\"answer\": \"(.*?)\"", text)
            if match:
                model_answer = match.group(1)
        
        if DEBUG and model_answer is None:
            print(output[-100:])
            pred = re.findall(r"```json.*?```|\{.*?\}", output, flags=re.DOTALL)
            print(pred)
            print("-----")

        
        if not model_answer:
            # skipping evaluation
            no_answer += 1
            parsed_item["model_answer"] = ""
            parsed_item["correct_answer"] = correct_answer
            parsed_item["matched"] = "No answer extracted"
            parsed_results.append(parsed_item) 
            continue
        

        
        correct = model_answer.strip().lower() == correct_answer.strip().lower()

        if DEBUG:
            print(correct, model_answer, correct_answer)
        
        if correct:
            solved_examples += 1


        reason_lens.append(len(reason))
        # parsed_item["reasoning"] = reason
        parsed_item["model_answer"] = model_answer
        parsed_item["correct_answer"] = correct_answer
        parsed_item["matched"] = correct
        parsed_results.append(parsed_item)

        category_correct[category] += correct
 
    result = {}
    result["Model"] = model.split("%")[0]
    result["Mode"] = model.split("%")[1]
    result["Acc"] = f"{solved_examples/num_total_examples*100:.2f}"
    result["No answer"] = f"{no_answer/num_total_examples*100:.2f}"
    result["Total"] = num_total_examples
    result["Reason Lens"] = f"{sum(reason_lens)/len(reason_lens):.2f}"
    result["Model"] = model_name_replacement(result["Model"])


    category_acc = {cat: f"{category_correct[cat]/category_total[cat]*100:.2f}" for cat in category_correct}

    return result, parsed_results, category_acc



def gen_results(task: MMLUProTask, run_name_folders: str): 

    model_results = load_model_results(run_name_folders)

    columns = ["Model", "Mode", "Acc", "No answer", "Total", "Reason Lens"]
    cats_names = []
    rows: List[dict] = []
    cats: List[dict] = []
    for model_name, filepath in model_results.items():       

        print(f"Processing {filepath}")

        # extract answer from generation and compare with ground truth
        result, parsed_results, categories_acc = eval_model(model_name, filepath) 

        for cat_name in categories_acc:
            if cat_name not in cats_names:
                cats_names.append(cat_name)


        # save the parsed_results to the same filepath with a  new prefix 
        parsed_results_filepath = filepath.replace("result_dirs", "result_dirs_parsed")
        # create folders if not exist
        os.makedirs(os.path.dirname(parsed_results_filepath), exist_ok=True)
        # save 
        with open(parsed_results_filepath, "w") as f:
            json.dump(parsed_results, f, indent=2)

        rows.append(result)
        cats.append(categories_acc)

    # sort the rows by overall accuracy
    idx_sort = sorted(range(len(rows)), key=lambda i: -float(rows[i]["Acc"]))
    rows = [rows[i] for i in idx_sort]
    cats = [cats[i] for i in idx_sort]
    cats_names.sort() # provides a consistent order
    
    # Convert rows to the expected format for tabulate
    table_data = [[row[col] for col in columns] + [cat[col] for col in cats_names] for row, cat in zip(rows,cats)]

    all_cols = columns + cats_names
    print(tabulate(table_data, headers=all_cols, tablefmt="fancy_outline", stralign="center", numalign="center"))

    data_name = task.name

    # write to markdown file
    with open(f"result_dirs/{data_name}.summary.md", "w") as f:
        f.write(tabulate(table_data, headers=all_cols, tablefmt="github", stralign="center", numalign="center"))

    # write to json file 
    with open(f"result_dirs/{data_name}.summary.json", "w") as f:
        json.dump(rows, f, indent=2)


if __name__ == "__main__":

    
    data_name = sys.argv[1]

    if data_name not in TASK_MAP:
        print(f"Invalid data name: {data_name}")
        sys.exit(1)

    task = TASK_MAP[data_name]

    run_name_folders = {
        "-": f"result_dirs/{data_name}", 
    }  

    gen_results(task, run_name_folders)
