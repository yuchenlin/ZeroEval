import json
from collections import defaultdict
import os
from tabulate import tabulate
from datasets import load_dataset

from .eval_utils import load_model_results, extract_last_complete_json, model_name_replacement

from collections import Counter
from collections import defaultdict
from src.tasks import TASKS_COLLECTION

TASK = TASKS_COLLECTION['zebra-grid']

private_solutions = {}

class NotEnoughExamplesError(Exception):
    pass

def load_private_solutions():
    global private_solutions
    private_zebra_data = load_dataset("allenai/ZebraLogicBench-private", "grid_mode", split="test")
    for item in private_zebra_data:
        private_solutions[item["id"]] = item["solution"]
    return


# Cache to store loaded data
file_cache = {}

def eval_model(model, filepath, mode="best_of_n", max_N=None):
    global private_solutions, file_cache

    # Check if the data is already cached
    if filepath in file_cache:
        print(f"Using cached data for {filepath}")
        data = file_cache[filepath]
    else:
        with open(filepath, "r") as f:
            print(f"Loading {filepath}")
            data = json.load(f)
            # Cache the loaded data
            file_cache[filepath] = data

    solved_puzzles = 0
    num_total_puzzles = len(data)
    correct_cells = 0
    total_cells = 0
    no_answer = 0

    
    if TASK.total_num_examples != num_total_puzzles:
        raise NotEnoughExamplesError("Task not Finished")

    num_total_puzzles_by_size = defaultdict(int)
    solved_puzzles_by_size = defaultdict(int)
    reason_lens = []
    parsed_results = []  # New list to store parsed results
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

        # Read and Parse the predictions from model output
        predictions = [extract_last_complete_json(output) for output in item["output"]]
        predictions = [p for p in predictions if p is not None and "solution" in p and p["solution"] is not None]

        # if all the predictions are empty, then skip the current puzzle, and add no answer count
        if not predictions:
            no_answer += 1
            # TODO: also add an item to the parsed results with no answer
            parsed_item = item.copy()
            parsed_item["reasoning"] = ""
            parsed_item["correct_cells"] = 0
            parsed_item["total_cells"] = this_total_cells
            parsed_item["solved"] = False
            parsed_item["parsed"] = False
            parsed_results.append(parsed_item)
            continue

        # Limit the number of predictions to max_N if specified
        if max_N is not None:
            predictions = predictions[:max_N]


        n_size = len(predictions)  # Capture the number of predictions



        if n_size == 1:
            mode = "single"
            # Single output case
            prediction_table = predictions[0]["solution"]
            reason = predictions[0].get("reasoning", "")
        elif mode == "rm_bon":
            # there are reward model scores in the output
            # the item contains two new keys: rm_scores, rm
            # rm_scores is a list of scores for each output in the same order
            # we should select the output with the highest score
            max_score = float("-inf")
            best_prediction = None
            for prediction, score in zip(predictions, item["rm_scores"]):
                if score > max_score:
                    max_score = score
                    best_prediction = prediction
            prediction_table = best_prediction["solution"]
            reason = best_prediction.get("reasoning", "")
        elif mode == "best_of_n":
            # Best of N: Choose the prediction with the maximum number of correct cells
            max_correct_cells = 0
            best_prediction = None
            for prediction in predictions:
                current_correct_cells = 0
                prediction_table = prediction["solution"]
                for house in solution_table:
                    for column in solution_table[house]:
                        if house in prediction_table and column in prediction_table[house]:
                            truth_cell = solution_table[house][column].lower().strip()
                            # Note that prediction_table[house][column] could be None
                            if prediction_table[house][column] is None:
                                continue
                            predicted_cell = prediction_table[house][column].lower().strip()
                            if truth_cell == predicted_cell:
                                current_correct_cells += 1
                if current_correct_cells > max_correct_cells or best_prediction is None:
                    max_correct_cells = current_correct_cells
                    best_prediction = prediction
            prediction_table = best_prediction["solution"]
            reason = best_prediction.get("reasoning", "")
        elif mode == "majority_of_n":
            # Majority of N: Perform majority voting for each cell
            prediction_table = {}
            for house in solution_table:
                prediction_table[house] = {}
                for column in solution_table[house]:
                    votes = []
                    for prediction in predictions:
                        if house in prediction["solution"] and column in prediction["solution"][house]:
                            predicted_cell = prediction["solution"][house][column]
                            if isinstance(predicted_cell, list):
                                predicted_cell = predicted_cell[0]
                            # Note that prediction_table[house][column] could be None
                            if predicted_cell is not None:
                                votes.append(predicted_cell.lower().strip())
                    if votes:
                        most_common = Counter(votes).most_common(1)[0][0]
                        prediction_table[house][column] = most_common
                    else:
                        prediction_table[house][column] = None
            # reason = ""  # Reasoning is not applicable for majority voting
            # use a random prediction to get the reasoning
            reason = predictions[0].get("reasoning", "")
        elif mode in ["most_common_of_n", "middle_common_of_n", "least_common_of_n"]:
            # Choose the prediction where the cell's value is the most common among all predictions at the same positions
            # Specifically, we give each value at each position a score based on its popularity, and the prediction with the highest sum of scores is chosen
            # Initialize a dictionary to store scores for each prediction
            prediction_scores = defaultdict(int)

            # Iterate over each house and column in the solution table
            for house in solution_table:
                for column in solution_table[house]:
                    # Count occurrences of each value a``````t the current position across all predictions
                    value_counter = Counter()
                    for prediction in predictions:
                        if house in prediction["solution"] and column in prediction["solution"][house]:
                            predicted_cell = prediction["solution"][house][column]
                            if isinstance(predicted_cell, list):
                                predicted_cell = predicted_cell[0]
                            if predicted_cell is not None:
                                value_counter[predicted_cell.lower().strip()] += 1

                    # Assign scores to each prediction based on the popularity of its value at the current position
                    for idx, prediction in enumerate(predictions):
                        if house in prediction["solution"] and column in prediction["solution"][house]:
                            predicted_cell = prediction["solution"][house][column]
                            if isinstance(predicted_cell, list):
                                predicted_cell = predicted_cell[0]
                            if predicted_cell is not None:
                                prediction_scores[idx] += value_counter[predicted_cell.lower().strip()]
            if mode == "most_common_of_n":
                # Select the prediction with the highest score
                best_index = max(range(len(predictions)), key=lambda idx: prediction_scores[idx])
                best_prediction = predictions[best_index]
                prediction_table = best_prediction["solution"]
                reason = best_prediction.get("reasoning", "")
            elif mode == "middle_common_of_n":
                # Select the prediction with the median score
                best_index = sorted(range(len(predictions)), key=lambda idx: prediction_scores[idx])[len(predictions) // 2]
                best_prediction = predictions[best_index]
                prediction_table = best_prediction["solution"]
                reason = best_prediction.get("reasoning", "")
            elif mode == "least_common_of_n":
                # Select the prediction with the lowest score
                best_index = min(range(len(predictions)), key=lambda idx: prediction_scores[idx])
                best_prediction = predictions[best_index]
                prediction_table = best_prediction["solution"]
                reason = best_prediction.get("reasoning", "")

        elif mode in ["longest_of_n", "shortest_of_n", "median_of_n"]:
            # Collect all predictions with their reasoning lengths
            predictions_with_lengths = [(prediction, len(prediction.get("reasoning", ""))) for prediction in predictions]

            # Sort by reasoning length
            predictions_with_lengths.sort(key=lambda x: x[1])

            if mode == "longest_of_n":
                best_prediction = predictions_with_lengths[-1][0]  # Last element for longest
            elif mode == "shortest_of_n":
                best_prediction = predictions_with_lengths[0][0]   # First element for shortest
            elif mode == "median_of_n":
                median_index = len(predictions_with_lengths) // 2
                best_prediction = predictions_with_lengths[median_index][0]  # Middle element for median

            prediction_table = best_prediction["solution"]
            reason = best_prediction.get("reasoning", "")

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
                    if truth_cell.lower().strip() == predicted_cell.lower().strip():
                        this_correct_cells += 1
        correct_cells += this_correct_cells

        # compute puzzle success rate
        if this_correct_cells == this_total_cells:
            solved_puzzles += 1
            solved_puzzles_by_size[size] += 1

        parsed_item = item.copy()
        parsed_item["reasoning"] = reason
        parsed_item["correct_cells"] = this_correct_cells
        parsed_item["total_cells"] = this_total_cells
        parsed_item["solved"] = this_correct_cells == this_total_cells
        parsed_item["parsed"] = True
        parsed_results.append(parsed_item)

    # # print the success rate by size; order the dict by size first
    sizes = sorted(num_total_puzzles_by_size.keys())
    easy_sizes =  ['2*2', '2*3', '2*4', '2*5', '2*6', '3*2', '3*3',]
    hard_sizes =  ['3*4', '3*5', '4*2', '3*6', '4*3', '4*4', '5*2', '6*2', '4*5', '4*6', '5*3', '5*4', '5*5', '5*6', '6*3', '6*4', '6*5', '6*6']

    small_sizes = ['2*2', '2*3', '2*4', '2*5', '2*6', '3*2', '3*3', '4*2']
    medium_sizes = ['3*4', '3*5', '3*6', '4*3', '4*4', '5*2', '6*2']
    large_sizes = ['4*5', '5*3', '4*6', '5*4', '6*3']
    xl_sizes = ['5*5', '6*4', '5*6', '6*5', '6*6']

    easy_solved_puzzles = sum([solved_puzzles_by_size[size] for size in easy_sizes])
    easy_total_puzzles = sum([num_total_puzzles_by_size[size] for size in easy_sizes])
    hard_solved_puzzles = sum([solved_puzzles_by_size[size] for size in hard_sizes])
    hard_total_puzzles = sum([num_total_puzzles_by_size[size] for size in hard_sizes])

    small_solved_puzzles = sum([solved_puzzles_by_size[size] for size in small_sizes])
    small_total_puzzles = sum([num_total_puzzles_by_size[size] for size in small_sizes])
    medium_solved_puzzles = sum([solved_puzzles_by_size[size] for size in medium_sizes])
    medium_total_puzzles = sum([num_total_puzzles_by_size[size] for size in medium_sizes])
    large_solved_puzzles = sum([solved_puzzles_by_size[size] for size in large_sizes])
    large_total_puzzles = sum([num_total_puzzles_by_size[size] for size in large_sizes])
    xl_solved_puzzles = sum([solved_puzzles_by_size[size] for size in xl_sizes])
    xl_total_puzzles = sum([num_total_puzzles_by_size[size] for size in xl_sizes])


    # for size in sizes:
        # print(f"Size {size}: {solved_puzzles_by_size[size]}/{num_total_puzzles_by_size[size]} -> {solved_puzzles_by_size[size]/num_total_puzzles_by_size[size]*100:.2f}%")

    result = {}
    result["Model"] = model.split("%")[0]
    result["Mode"] = model.split("%")[1]
    result["Puzzle Acc"] = f"{solved_puzzles/num_total_puzzles*100:.2f}"
    result["Cell Acc"] = f"{correct_cells/total_cells*100:.2f}"
    result["No answer"] = f"{no_answer/num_total_puzzles*100:.2f}"
    result["Easy Puzzle Acc"] = f"{easy_solved_puzzles/easy_total_puzzles*100:.2f}"
    result["Hard Puzzle Acc"] = f"{hard_solved_puzzles/hard_total_puzzles*100:.2f}"
    result["Small Puzzle Acc"] = f"{small_solved_puzzles/small_total_puzzles*100:.2f}"
    result["Medium Puzzle Acc"] = f"{medium_solved_puzzles/medium_total_puzzles*100:.2f}"
    result["Large Puzzle Acc"] = f"{large_solved_puzzles/large_total_puzzles*100:.2f}"
    result["XL Puzzle Acc"] = f"{xl_solved_puzzles/xl_total_puzzles*100:.2f}"
    result["Total Puzzles"] = num_total_puzzles
    result["Reason Lens"] = f"{sum(reason_lens)/len(reason_lens):.2f}"
    result["Model"] = model_name_replacement(result["Model"])
    result["N_Mode"] = "single" if n_size == 1 else mode
    result["N_Size"] = n_size
    return result, parsed_results  # Return parsed_results along with the result


def gen_results(run_name_folders, bon=False, save_results=True):
    model_results = load_model_results(run_name_folders)

    def save_parsed_results(filepath, parsed_results, bon=bon):
        parsed_results_filepath = filepath.replace("result_dirs", "result_dirs_parsed")
        # Create folders if they don't exist
        os.makedirs(os.path.dirname(parsed_results_filepath), exist_ok=True)
        if bon:
            # remove the outputs from the parsed results
            parsed_results_no_output = []
            for parsed_item in parsed_results:
                parsed_item_no_output = parsed_item.copy()
                if "output" in parsed_item_no_output:
                    del parsed_item_no_output["output"]
                parsed_results_no_output.append(parsed_item_no_output)
            parsed_results = parsed_results_no_output
        # Save parsed results
        with open(parsed_results_filepath, "w") as f:
            json.dump(parsed_results, f, indent=2)
            print(f"Saved to {f.name}")

    # columns = ["Model", "Mode", "N_Mode", "N_Size", "Puzzle Acc", "Easy Puzzle Acc", "Hard Puzzle Acc", "Cell Acc",  "No answer",  "Total Puzzles", "Reason Lens"]
    columns = ["Model", "Mode", "N_Mode", "N_Size", "Puzzle Acc", "Small Puzzle Acc", "Medium Puzzle Acc", "Large Puzzle Acc", "XL Puzzle Acc", "Cell Acc",  "No answer",  "Total Puzzles", "Reason Lens"]
    rows = []

    for model_name, filepath in model_results.items():


        if bon and "bon_" in filepath or "rm_" in filepath:
            # result, parsed_results = eval_model(model_name, filepath, mode="majority_of_n", max_N=32)
            # result, parsed_results = eval_model(model_name, filepath, mode="most_common_of_n", max_N=64)
            # result, parsed_results = eval_model(model_name, filepath, mode="longest_of_n", max_N=32)
            # result, parsed_results = eval_model(model_name, filepath, mode="shortest_of_n", max_N=32)
            # result, parsed_results = eval_model(model_name, filepath, mode="median_of_n", max_N=32)
            # result, parsed_results = eval_model(model_name, filepath, mode="least_common_of_n", max_N=32)
            # result, parsed_results = eval_model(model_name, filepath, mode="middle_common_of_n", max_N=32)
            for K in [1, 4, 8, 16, 32, 64, 128, 256]:
                if "gpt-4o-2024-08-06" in model_name and K > 128:
                    continue
                if "rm_32" in filepath and K > 32:
                    continue

                result, parsed_results = eval_model(model_name, filepath, mode="rm_bon", max_N=K)

                save_parsed_results(filepath.replace(".json", f".rm_bon.K={K}.json"), parsed_results)
                rows.append(result)

                # result, parsed_results = eval_model(model_name, filepath, mode="best_of_n", max_N=K)
                # save_parsed_results(filepath.replace(".json", f".best_of_n.K={K}.json"), parsed_results)
                # rows.append(result)

                # result, parsed_results = eval_model(model_name, filepath, mode="majority_of_n", max_N=K)
                # save_parsed_results(filepath.replace(".json", f".majority_of_n.K={K}.json"), parsed_results)
                # rows.append(result)

                # result, parsed_results = eval_model(model_name, filepath, mode="most_common_of_n", max_N=K)
                # save_parsed_results(filepath.replace(".json", f".most_common_of_n.K={K}.json"), parsed_results)
                # rows.append(result)
        else:
            # Save the parsed_results to the same filepath with a new prefix
            try:
                result, parsed_results = eval_model(model_name, filepath, mode="single")
            except NotEnoughExamplesError:
                print("Not enough generations. skipping.")
                continue
            
            save_parsed_results(filepath, parsed_results)
            rows.append(result)

    if "bon_" in filepath:
        # rows = sorted(rows, key=lambda x: -float(x["Puzzle Acc"]))
        # first sort by N_Mode and then sort by N_Size
        # Sort the rows first by model name and then "N_Mode" and then by "N_Size"
        rows = sorted(rows, key=lambda x: (x["Model"], x["N_Mode"], x["N_Size"]))
    else:
        # sort the rows by puzzle accuracy
        rows = sorted(rows, key=lambda x: -float(x["Puzzle Acc"]))
    # Convert rows to the expected format for tabulate
    table_data = [[row[col] for col in columns] for row in rows]

    print(tabulate(table_data, headers=columns, tablefmt="fancy_outline", stralign="center", numalign="center"))
    # print(tabulate(rows, headers=columns, tablefmt="github"))

    if save_results:
        # write to json file
        with open("result_dirs/zebra-grid.summary.json", "w") as f:
            json.dump(rows, f, indent=2)

        # write to markdown file
        with open(f"result_dirs/zebra-grid.summary.md", "w") as f:
            f.write(tabulate(table_data, headers=columns, tablefmt="github", stralign="center", numalign="center"))



if __name__ == "__main__":
    
    run_name_folders = {
        "greedy": "result_dirs/zebra-grid",
        # "sampling": "result_dirs/zebra-grid/sampling",
        # "bon_all": "result_dirs/zebra-grid/bon_all",
        # "rm": "result_dirs/zebra-grid/rm_32",
        # "self_verification": "result_dirs/zebra-grid/self_verification/",
        # "neg_feedback": "result_dirs/zebra-grid/neg_feedback_v2",
        # "zebra_oracle": "result_dirs/zebra-grid/zebra_oracle/",
    }
    load_private_solutions()
    gen_results(run_name_folders, bon=False, save_results=True)

