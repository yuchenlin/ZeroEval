import json 
from collections import defaultdict
import os 
from tabulate import tabulate 
import re
import sys 
from src.evaluation.eval_utils import load_model_results, extract_values_from_json, extract_first_complete_json, model_specific_extraction, model_name_replacement
from src.tasks import HendrycksMathTask
from src.evaluation.hendrycks_math_utils import is_equiv
import re

DEBUG = False
TASK = HendrycksMathTask()

def sanitize_math_answers(answer):
    # ignore symbols like $ 
    answer = answer.replace("$", "").strip()
    # ignore the units like miles after the number  
    # remove "," in the number
    answer = answer.replace(",", "")
    # convert fractions to float
    if "/" in answer:
        try:
            answer = str(float(eval(answer)))
        except:
            pass
    return answer



def eval_model(model, filepath):
    global private_solutions

    with open(filepath, "r") as f:
        print(f"Processing {filepath}")
        data = json.load(f)

    solved_examples = 0 
    num_total_examples = len(data) 
    no_answer = 0  
    
    reason_lens = []
    parsed_results = [] 
    for item in data:  

        # Read and Parse the prediction from model output
        parsed_item = item.copy()
        output = item["output"][0]
        reason = output # as proxy full output with reasoning

        predictions = re.findall("```json(.*?)```", output, flags=re.DOTALL)

        model_answer = None
        if predictions:
            text = predictions[-1]
            text = re.sub(r'\\\\', r'\\', text)
            match = re.search("\"answer\": \"[$](.*?)[$]\"", text)
            if match:
                model_answer = match.group(1)
    
        correct_answer = item["answer"]

        if not model_answer:
            # skipping evaluation
            no_answer += 1
            parsed_item["model_answer"] = ""
            parsed_item["correct_answer"] = correct_answer
            parsed_item["matched"] = "No answer extracted"
            parsed_results.append(parsed_item) 

            continue
        
        
        correct = is_equiv(model_answer, correct_answer)
        if DEBUG:
            print(correct, model_answer, correct_answer)
        
        if correct:
            solved_examples += 1


        reason_lens.append(len(reason))

        parsed_item["reasoning"] = reason
        parsed_item["model_answer"] = model_answer
        parsed_item["correct_answer"] = correct_answer
        parsed_item["matched"] = correct
        parsed_results.append(parsed_item)


 
    result = {}
    result["Model"] = model.split("%")[0]
    result["Mode"] = model.split("%")[1]
    result["Acc"] = f"{solved_examples/num_total_examples*100:.2f}"
    result["No answer"] = f"{no_answer/num_total_examples*100:.2f}"
    result["Total"] = num_total_examples
    result["Reason Lens"] = f"{sum(reason_lens)/len(reason_lens):.2f}"
    result["Model"] = model_name_replacement(result["Model"])
    return result, parsed_results


def gen_results(run_name_folders): 
    model_results = load_model_results(run_name_folders)


    columns = ["Model", "Mode", "Acc", "No answer", "Total", "Reason Lens"]
    rows = []
    for model_name, filepath in model_results.items():       
        if DEBUG:
            print("")
            print("")
            print("")
            print("")
            print(model_name)
        # extract answer from generation and compare with ground truth
        result, parsed_results = eval_model(model_name, filepath) 

        # save the parsed_results to the same filepath with a  new prefix 
        parsed_results_filepath = filepath.replace("result_dirs", "result_dirs_parsed")
        # create folders if not exist
        os.makedirs(os.path.dirname(parsed_results_filepath), exist_ok=True)
        # save 
        with open(parsed_results_filepath, "w") as f:
            json.dump(parsed_results, f, indent=2)
        rows.append(result)

    # sort the rows by puzzle accuracy
    rows = sorted(rows, key=lambda x: -float(x["Acc"]))
    # Convert rows to the expected format for tabulate
    table_data = [[row[col] for col in columns] for row in rows]

    print(tabulate(table_data, headers=columns, tablefmt="fancy_outline", stralign="center", numalign="center"))

    data_name = TASK.name

    # write to markdown file
    with open(f"result_dirs/{data_name}.summary.md", "w") as f:
        f.write(tabulate(table_data, headers=columns, tablefmt="github", stralign="center", numalign="center"))

    # write to json file 
    with open(f"result_dirs/{data_name}.summary.json", "w") as f:
        json.dump(rows, f, indent=2)


if __name__ == "__main__":

    

    run_name_folders = {
        "-": TASK.get_output_dir()
    }  

    gen_results(run_name_folders)
