import json 
from collections import defaultdict
import os 
from tabulate import tabulate

# filepath = "result_dirs/zebra-grid/Meta-Llama-3-70B-Instruct.json"
# filepath = "result_dirs/zebra-grid/Meta-Llama-3-8B-Instruct.json"
# filepath = "result_dirs/zebra-grid/gpt-3.5-turbo-0125.json"
# filepath = "result_dirs/zebra-grid/claude-3-haiku-20240307.json"
# filepath = "result_dirs/zebra-grid/claude-3-5-sonnet-20240620.json"
# filepath = "result_dirs/zebra-grid/claude-3-opus-20240229.json"
# filepath = "result_dirs/zebra-grid/claude-3-sonnet-20240229.json"
# filepath = "result_dirs/zebra-grid/gpt-4o-2024-05-13.json"
# filepath = "result_dirs/zebra-grid/gpt-4-turbo-2024-04-09.json"
# filepath = "result_dirs/zebra-grid/gemini-1.5-pro.json"
# filepath = "result_dirs/zebra-grid/gemini-1.5-flash.json"
# filepath = "result_dirs/zebra-grid/reka-flash-20240226.json"
# filepath = "result_dirs/zebra-grid/reka-core-20240501.json"
# filepath = "result_dirs/zebra-grid/Qwen2-72B-Instruct.json"


folder = "result_dirs/zebra-grid/cot=True"

model_results = {}

# iterate all json files under the folder 
for filename in os.listdir(folder):
    filepath = os.path.join(folder, filename)
    if not filename.endswith(".json"):
        continue
    model_results[filename.replace(".json", "")] = filepath  
 
def extract_last_complete_json(s):
    # Stack to keep track of opening and closing braces
    stack = []
    last_json_start = None
    last_json_str = None
    
    for i, char in enumerate(s):
        if char == '{':
            stack.append(i)
            if last_json_start is None:
                last_json_start = i
        elif char == '}':
            if stack:
                start = stack.pop()
                if not stack:
                    # Complete JSON object found
                    last_json_str = s[last_json_start:i+1]
                    last_json_start = None
    
    # Load the last JSON object
    if last_json_str:
        try:
            return json.loads(last_json_str)
        except json.JSONDecodeError:
            pass
    
    return None

def eval_model(model, filepath):
    with open(filepath, "r") as f:
        data = json.load(f)

    solved_puzzles = 0 
    num_total_puzzles = len(data)
    correct_cells = 0
    total_cells = 0
    no_asnwer = 0 

    num_total_puzzles_by_size = defaultdict(int)
    solved_puzzles_by_size = defaultdict(int) 

    for item in data:
        solution = item["solution"]
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
        prediction_table = extract_last_complete_json(prediction_str)
        if prediction_table is None:
            # print(prediction_str)
            no_asnwer += 1
            # print(item["id"])
            continue 


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
        
        # compute puzzle-level success rate
        if this_correct_cells == this_total_cells:
            solved_puzzles += 1
            solved_puzzles_by_size[size] += 1
            
    # print(f"Filepath: {filepath}")
    # print(f"Puzzle-level success rate: {solved_puzzles}/{num_total_puzzles} -> {solved_puzzles/num_total_puzzles*100:.2f}%")
    # print(f"Cell-wise accuracy: {correct_cells}/{total_cells}  -> {correct_cells/total_cells*100:.2f}%")
    # print(f"No answer: {no_asnwer} -> {no_asnwer/num_total_puzzles*100:.2f}%")

    # # print the success rate by size; order the dict by size first 
    easy_sizes = ["3*3", "3*4", "3*5", "3*6", "4*3"]
    hard_sizes = ['4*4', '4*5', '4*6', '5*3', '5*4', '5*5', '5*6', '6*3', '6*4', '6*5', '6*6']
    sizes = sorted(num_total_puzzles_by_size.keys())
    # print(sizes) --> ['3*3', '3*4', '3*5', '3*6', '4*3', '4*4', '4*5', '4*6', '5*3', '5*4', '5*5', '5*6', '6*3', '6*4', '6*5', '6*6']
    easy_solved_puzzles = sum([solved_puzzles_by_size[size] for size in easy_sizes])
    easy_total_puzzles = sum([num_total_puzzles_by_size[size] for size in easy_sizes])
    hard_solved_puzzles = sum([solved_puzzles_by_size[size] for size in hard_sizes])
    hard_total_puzzles = sum([num_total_puzzles_by_size[size] for size in hard_sizes])

    # for size in sizes:
        # print(f"Size {size}: {solved_puzzles_by_size[size]}/{num_total_puzzles_by_size[size]} -> {solved_puzzles_by_size[size]/num_total_puzzles_by_size[size]*100:.2f}%")

    result = {}
    result["Model"] = model
    result["Puzzle-level Acc"] = f"{solved_puzzles/num_total_puzzles*100:.2f}"
    result["Cell-wise Acc"] = f"{correct_cells/total_cells*100:.2f}"
    result["No answer"] = f"{no_asnwer/num_total_puzzles*100:.2f}"
    result["Easy Puzzle-level Acc"] = f"{easy_solved_puzzles/easy_total_puzzles*100:.2f}"
    result["Hard Puzzle-level Acc"] = f"{hard_solved_puzzles/hard_total_puzzles*100:.2f}"
    return result


columns = ["Model", "Puzzle-level Acc", "Cell-wise Acc", "No answer", "Easy Puzzle-level Acc", "Hard Puzzle-level Acc"]
rows = []
for model_name, filepath in model_results.items():
    # print(f"Model: {model_name}")
    result = eval_model(model_name, filepath)
    # print(json.dumps(data, indent=2))
    rows.append(result)

# sort the rows by puzzle-level accuracy
rows = sorted(rows, key=lambda x: -float(x["Puzzle-level Acc"]))
# Convert rows to the expected format for tabulate
table_data = [[row[col] for col in columns] for row in rows]

print(tabulate(table_data, headers=columns, tablefmt="fancy_outline"))
# print(tabulate(rows, headers=columns, tablefmt="github"))

    
