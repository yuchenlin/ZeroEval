import json 
from collections import defaultdict
import os 
from tabulate import tabulate
from datasets import load_dataset

from eval_utils import load_model_results, extract_last_complete_json

private_solutions = {}

def load_private_solutions():
    global private_solutions
    private_zebra_data = load_dataset("allenai/ZebraLogicBench-private", "grid_mode", split="test")
    for item in private_zebra_data:
        private_solutions[item["id"]] = item["solution"] 
    return 



def eval_model(model, filepath):
    global private_solutions
    with open(filepath, "r") as f:
        print(f"Processing {filepath}")
        data = json.load(f)

    solved_puzzles = 0 
    num_total_puzzles = len(data)
    correct_cells = 0
    total_cells = 0
    no_asnwer = 0 

    num_total_puzzles_by_size = defaultdict(int)
    solved_puzzles_by_size = defaultdict(int) 
    reason_lens = []
    for item in data:
        # solution = item["solution"]
        solution = private_solutions[item["id"]]
        size = item["size"]
        num_total_puzzles_by_size[size] += 1

        # Process the solution 
        solution_table = {}
        num_houses = len(solution["rows"])
        columns = solution["header"]
        assert columns[0] == "House"
        solution_table = {}
        this_total_cells = 0 
        for i in range(num_houses):
            solution_table[f'House {i+1}'] = {columns[j]: solution["rows"][i][j] for j in range(1, len(columns))} 
            this_total_cells += len(columns) - 1
        total_cells += this_total_cells

        # Read and Parse the prediction from model output
        prediction_str = item["output"][0]     
        prediction_json = extract_last_complete_json(prediction_str)
        if prediction_json is None or "solution" not in prediction_json or prediction_json["solution"] is None:
            # print("-"*100)
            # prediction_str = prediction_str.replace("\n", "")
            # print([prediction_str])
            # json.loads(prediction_str)
            no_asnwer += 1
            # print(item["id"])
            continue 
        reason = prediction_json.get("reasoning", "")
        prediction_table = prediction_json["solution"]
        
        reason_lens.append(len(reason))

        this_correct_cells = 0 # number in the solution_table 
        for house in solution_table:
            for column in solution_table[house]: 
                # if prediction_table[house][column] not exist then pass 
                if house in prediction_table and column in prediction_table[house]:
                    truth_cell = solution_table[house][column].lower().strip()
                    if prediction_table[house][column] is None or len(prediction_table[house][column]) == 0:
                        continue
                    if type(prediction_table[house][column]) == list:
                        predicted_cell = prediction_table[house][column][0].lower().strip()
                    elif type(prediction_table[house][column]) == str:
                        predicted_cell = prediction_table[house][column].lower().strip()
                    else:
                        raise ValueError(f"Unknown type: {type(prediction_table[house][column])}")
                    if truth_cell == predicted_cell:
                        this_correct_cells += 1  
        correct_cells += this_correct_cells
        
        # compute puzzle success rate
        if this_correct_cells == this_total_cells:
            solved_puzzles += 1
            solved_puzzles_by_size[size] += 1

        
         

    # # print the success rate by size; order the dict by size first  
    sizes = sorted(num_total_puzzles_by_size.keys()) 
    easy_sizes =  ['2*2', '2*3', '2*4', '2*5', '2*6', '3*2', '3*3',] 
    hard_sizes =  ['3*4', '3*5', '4*2', '3*6', '4*3', '4*4', '5*2', '6*2', '4*5', '4*6', '5*3', '5*4', '5*5', '5*6', '6*3', '6*4', '6*5', '6*6']
    
    easy_solved_puzzles = sum([solved_puzzles_by_size[size] for size in easy_sizes])
    easy_total_puzzles = sum([num_total_puzzles_by_size[size] for size in easy_sizes]) 
    hard_solved_puzzles = sum([solved_puzzles_by_size[size] for size in hard_sizes])
    hard_total_puzzles = sum([num_total_puzzles_by_size[size] for size in hard_sizes])

    # for size in sizes:
        # print(f"Size {size}: {solved_puzzles_by_size[size]}/{num_total_puzzles_by_size[size]} -> {solved_puzzles_by_size[size]/num_total_puzzles_by_size[size]*100:.2f}%")

    result = {}
    result["Model"] = model.split("%")[0]
    result["Mode"] = model.split("%")[1]
    result["Puzzle Acc"] = f"{solved_puzzles/num_total_puzzles*100:.2f}"
    result["Cell Acc"] = f"{correct_cells/total_cells*100:.2f}"
    result["No answer"] = f"{no_asnwer/num_total_puzzles*100:.2f}"
    result["Easy Puzzle Acc"] = f"{easy_solved_puzzles/easy_total_puzzles*100:.2f}" 
    result["Hard Puzzle Acc"] = f"{hard_solved_puzzles/hard_total_puzzles*100:.2f}"
    result["Total Puzzles"] = num_total_puzzles
    result["Reason Lens"] = f"{sum(reason_lens)/len(reason_lens):.2f}"
    return result


def gen_results(run_name_folders): 
    model_results = load_model_results(run_name_folders)

    columns = ["Model", "Mode", "Puzzle Acc", "Cell Acc", "No answer", "Easy Puzzle Acc", "Hard Puzzle Acc", "Total Puzzles", "Reason Lens"]
    rows = []
    for model_name, filepath in model_results.items(): 
        result = eval_model(model_name, filepath) 
        rows.append(result)

    # sort the rows by puzzle accuracy
    rows = sorted(rows, key=lambda x: -float(x["Puzzle Acc"]))
    # Convert rows to the expected format for tabulate
    table_data = [[row[col] for col in columns] for row in rows]

    print(tabulate(table_data, headers=columns, tablefmt="fancy_outline", stralign="center", numalign="center"))
    # print(tabulate(rows, headers=columns, tablefmt="github"))

    # write to json file 
    with open("result_dirs/zebra-grid.summary.json", "w") as f:
        json.dump(rows, f, indent=2)

    # write to markdown file
    # with open(f"result_dirs/zebra-grid.summary.md", "w") as f:
    #     f.write(tabulate(table_data, headers=columns, tablefmt="github", stralign="center", numalign="center"))


if __name__ == "__main__":
    run_name_folders = {
        "greedy": "result_dirs/zebra-grid",
        "sampling": "result_dirs/zebra-grid/sampling",
    } 
    load_private_solutions()
    gen_results(run_name_folders)
