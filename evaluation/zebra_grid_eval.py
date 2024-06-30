import json 
import re 


filepath = "result_dirs/zebra-grid/Meta-Llama-3-70B-Instruct.json"


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


with open(filepath, "r") as f:
    data = json.load(f)



solved_puzzles = 0 
num_total_puzzles = len(data)
correct_cells = 0
total_cells = 0
no_asnwer = 0 

for item in data:
    solution = item["solution"]
    prediction_str = item["output"][0]  
    
    prediction_table = extract_last_complete_json(prediction_str)
    if prediction_table is None:
        # print(prediction_str)
        no_asnwer += 1
        continue 

    solution_table = {}
    num_houses = len(solution["rows"])
    columns = solution["header"]
    assert columns[0] == "House"
    solution_table = {}
    for i in range(num_houses):
        solution_table[f'House {i+1}'] = {columns[j]: solution["rows"][i][j] for j in range(1, len(columns))}
    # print(json.dumps(prediction_table, indent=4))
    # print(json.dumps(solution_table, indent=4))
    # compute cell-wise accuracy; meaning that how many cells are correct
    this_correct_cells = 0
    this_total_cells = 0 # number in the solution_table 
    for house in solution_table:
        for column in solution_table[house]:
            this_total_cells += 1
            # if prediction_table[house][column] not exist then pass 
            if house in prediction_table and column in prediction_table[house]:
                truth_cell = solution_table[house][column].lower().strip()
                if prediction_table[house][column] is None:
                    continue
                if type(prediction_table[house][column]) == list:
                    predicted_cell = prediction_table[house][column][0].lower().strip()
                else:
                    predicted_cell = prediction_table[house][column].lower().strip()
                
                if truth_cell == predicted_cell:
                    this_correct_cells += 1  
    correct_cells += this_correct_cells
    total_cells += this_total_cells
    # compute puzzle-level success rate
    if this_correct_cells == this_total_cells:
        solved_puzzles += 1

print(f"Puzzle-level success rate: {solved_puzzles}/{num_total_puzzles}")
print(f"Cell-wise accuracy: {correct_cells}/{total_cells}")
print(f"No answer: {no_asnwer}")

    